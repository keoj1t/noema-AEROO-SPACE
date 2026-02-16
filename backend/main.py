from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import onnxruntime as ort
import json
from datetime import datetime, timezone
import io
import os
import tempfile
import uuid
from pathlib import Path
import traceback


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


from chronos_infer import detect_chronos_anomalies
from yolo_infer import detect_visual_anomalies, YOLOAnomalyDetector
from voice import VoiceAnalyzeConfig, analyze_voice_bytes

app = FastAPI(
    title="NOEMA API",
    description="AI System for Satellite Telemetry Analysis",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "models/model.onnx"
session = None


import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
abs_model_path = os.path.join(current_dir, MODEL_PATH)

print(f"📍 Current directory: {current_dir}")
print(f"📍 Searching for model at: {abs_model_path}")

try:
    if os.path.exists(abs_model_path):
        session = ort.InferenceSession(abs_model_path, providers=['CPUExecutionProvider'])
        print(f"✓ ONNX model loaded successfully: {abs_model_path}")
        print(f"✓ Model inputs: {[input.name for input in session.get_inputs()]}")
        print(f"✓ Model outputs: {[output.name for output in session.get_outputs()]}")
    elif os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print(f"✓ ONNX модель загружена успешно: {MODEL_PATH}")
    else:
        print(f"⚠ Модель не найдена по пути: {abs_model_path}")
        print(f"⚠ Искали также: {MODEL_PATH}")
        print("⚠ Используем аналитику на основе правил")
except Exception as e:
    print(f"⚠ Ошибка загрузки модели: {e}")
    import traceback
    traceback.print_exc()
    print("⚠ Используем аналитику на основе правил")


frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")


analyses_db = {}


CHRONOS_API_KEY = os.getenv('CHRONOS_API_KEY')
YOLO_API_KEY = os.getenv('YOLO_API_KEY')
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')

def make_api_key_dependency(env_var_name: str):
    expected = os.getenv(env_var_name)
    async def _verify(x_api_key: Optional[str] = Header(None)):
        if expected:
            if not x_api_key or x_api_key != expected:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return True
    return _verify

chronos_api_dep = make_api_key_dependency('CHRONOS_API_KEY')
yolo_api_dep = make_api_key_dependency('YOLO_API_KEY')


class AnalysisRequest(BaseModel):
    filename: str
    file_content: str  

class AgentQuery(BaseModel):
    question: str
    analysis_id: str


class YOLOFrameRequest(BaseModel):
    image: str
    analysis_id: Optional[str] = None

def preprocess_telemetry(df: pd.DataFrame) -> np.ndarray:
    """Preprocess telemetry data"""
    numeric_cols = ['battery_temp', 'voltage', 'orientation_error']
    
    
    df_processed = df.copy()
    comm_mapping = {'OK': 0.0, 'DEGRADED': 0.5, 'LOST': 1.0}
    df_processed['comm_status_encoded'] = df_processed['comm_status'].map(
        lambda x: comm_mapping.get(x, 0.0)
    )
    
    df_processed = df_processed.ffill().bfill()
    
    features = df_processed[numeric_cols + ['comm_status_encoded']].values
    
    return features.astype(np.float32)

def compute_anomaly_score_with_model(features: np.ndarray) -> np.ndarray:
    if session is None:
        return compute_anomaly_score_heuristic(features)
    
    try:
        input_name = session.get_inputs()[0].name
        reconstruction = session.run(None, {input_name: features})[0]
        errors = np.mean((features - reconstruction) ** 2, axis=1)
        
        return errors
    except Exception as e:
        print(f"⚠ Ошибка инференса модели: {e}")
        return compute_anomaly_score_heuristic(features)

def compute_anomaly_score_heuristic(features: np.ndarray) -> np.ndarray:
    """Heuristic anomaly calculation"""
    median_vals = np.median(features, axis=0)
    errors = np.mean(np.abs(features - median_vals), axis=1)
    errors = errors / (np.max(errors) + 1e-8)
    
    return errors

def detect_anomalies(anomaly_scores: np.ndarray, threshold_std: float = 1.5) -> Dict[str, Any]:
    """Detect anomalies in time series"""
    if len(anomaly_scores) < 5:
        return {
            "first_anomaly": -1,
            "baseline_mean": np.mean(anomaly_scores),
            "baseline_std": np.std(anomaly_scores),
            "anomaly_threshold": np.mean(anomaly_scores) + threshold_std * np.std(anomaly_scores)
        }
    
    # Для маленьких датасетов используем более чувствительный подход
    baseline_len = min(3, len(anomaly_scores) - 2)
    baseline_mean = np.mean(anomaly_scores[:baseline_len])
    baseline_std = np.std(anomaly_scores[:baseline_len]) if np.std(anomaly_scores[:baseline_len]) > 0 else 1
    
    threshold = baseline_mean + threshold_std * baseline_std
    
    # Ищем точку, где anomaly резко выскочила
    first_anomaly = -1
    for i in range(len(anomaly_scores) - 1):
        if anomaly_scores[i] > threshold:
            first_anomaly = i
            break
    
    return {
        "first_anomaly": first_anomaly,
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "anomaly_threshold": float(threshold)
    }

def analyze_root_cause(df: pd.DataFrame, anomaly_idx: int) -> tuple:
    """Analyze probable root cause"""
    if len(df) == 0:
        return "Insufficient data", []
    
    last_row = df.iloc[-1]
    
    if last_row['comm_status'] == 'LOST':
        if last_row['voltage'] < 3.0:
            root_cause = "Critical power voltage loss"
            causal_chain = [
                "Battery/solar panel degradation",
                "Voltage drop below critical level",
                "Power systems failure",
                "Loss of orientation",
                "Communication disruption"
            ]
        elif last_row['orientation_error'] > 30.0:
            root_cause = "Critical orientation error"
            causal_chain = [
                "Gyroscope/reaction wheel failure",
                "Uncontrolled rotation",
                "Antenna aiming impossible",
                "Loss of Earth contact"
            ]
        elif last_row['battery_temp'] > 60.0:
            root_cause = "Thermal system overheat"
            causal_chain = [
                "Thermal regulation system failure",
                "Critical component overheating",
                "Automatic system shutdown",
                "Communication loss"
            ]
        else:
            root_cause = "Communication system failure"
            causal_chain = [
                "Transmitter/receiver failure",
                "Signal loss",
                "Command reception failure"
            ]
    elif last_row['comm_status'] == 'DEGRADED':
        root_cause = "Partial system degradation"
        causal_chain = [
            "Initial failure stage",
            "Parameter degradation",
            "Overall reliability reduction"
        ]
    else:
        root_cause = "Normal operation with temporary anomalies"
        causal_chain = [
            "Temporary telemetry deviations",
            "Automatic compensation",
            "Stable operation"
        ]
    
    return root_cause, causal_chain

def generate_counterfactual(root_cause: str) -> str:
    """Generate counterfactual explanation"""
    counterfactuals = {
        "Critical power voltage loss": 
            "If the voltage monitoring system had triggered 15 minutes earlier, "
            "it would have been possible to switch to backup power and avoid complete failure.",
        
        "Critical orientation error":
            "With an additional standby gyroscope and immediate switching, "
            "orientation could have been restored before losing communication.",
        
        "Thermal system overheat":
            "Activating backup radiators and reducing power consumption by 30% "
            "could have prevented thermal escalation.",
        
        "Communication system failure":
            "Regular antenna calibration and amplifier checks could have prevented "
            "sudden communication system failure.",
        
        "Partial system degradation":
            "Scheduled maintenance and replacement of degrading components before critical "
            "levels would have maintained system operability.",
        
        "Normal operation with temporary anomalies":
            "Current monitoring system configuration adequately handles "
            "temporary deviations."
    }
    
    return counterfactuals.get(root_cause, 
        "Early anomaly detection and preventive measures could have prevented failure escalation.")

def analyze_telemetry_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Основная функция анализа телеметрии"""
    
    if len(df) < 5:
        raise ValueError("Недостаточно данных для анализа (минимум 5 строк)")
    
    # Предобработка
    features = preprocess_telemetry(df)
    
    # Вычисление аномалий
    anomaly_scores = compute_anomaly_score_with_model(features)
    
    # Конвертация в health scores (0-100)
    # Нормализуем аномалии с использованием z-score
    anomaly_mean = np.mean(anomaly_scores)
    anomaly_std = np.std(anomaly_scores) + 1e-8
    
    # Z-score нормализация
    normalized_anomalies = (anomaly_scores - anomaly_mean) / anomaly_std
    
    # Преобразуем в health (0-100 диапазон)
    # Высокие аномалии = низкое здоровье
    health_scores = 100 / (1 + np.exp(normalized_anomalies))  # Sigmoid
    
    # Клипируем в диапазон 0-100
    health_scores = np.clip(health_scores, 0, 100)
    
    # Обнаружение аномалий
    anomaly_info = detect_anomalies(anomaly_scores)
    
    # Анализ первопричины
    root_cause, causal_chain = analyze_root_cause(df, anomaly_info["first_anomaly"])
    
    # Формирование временных меток
    if anomaly_info["first_anomaly"] != -1 and anomaly_info["first_anomaly"] < len(df):
        first_anomaly_time = df.iloc[anomaly_info["first_anomaly"]]['time']
    else:
        first_anomaly_time = "N/A"
    
    # Расчет метрик
    initial_health = float(health_scores[0]) if len(health_scores) > 0 else 100.0
    final_health = float(health_scores[-1]) if len(health_scores) > 0 else 100.0
    degradation_rate = abs(final_health - initial_health) / (initial_health if initial_health > 0 else 1) * 100
    
    return {
        "health_scores": health_scores.tolist(),
        "anomaly_scores": anomaly_scores.tolist(),
        "timestamps": df['time'].astype(str).tolist(),
        "first_anomaly_index": int(anomaly_info["first_anomaly"]),
        "first_anomaly_time": str(first_anomaly_time),
        "root_cause": root_cause,
        "causal_chain": causal_chain,
        "final_health": final_health,
        "initial_health": initial_health,
        "degradation_rate": float(degradation_rate),
        "metrics": {
            "voltage_stats": {
                "min": float(df['voltage'].min()),
                "max": float(df['voltage'].max()),
                "mean": float(df['voltage'].mean()),
                "final": float(df['voltage'].iloc[-1])
            },
            "temp_stats": {
                "min": float(df['battery_temp'].min()),
                "max": float(df['battery_temp'].max()),
                "mean": float(df['battery_temp'].mean()),
                "final": float(df['battery_temp'].iloc[-1])
            },
            "orientation_stats": {
                "min": float(df['orientation_error'].min()),
                "max": float(df['orientation_error'].max()),
                "mean": float(df['orientation_error'].mean()),
                "final": float(df['orientation_error'].iloc[-1])
            }
        }
    }

def generate_3d_visualization_data(analysis: Dict[str, Any], 
                                   chronos_results: Dict[str, Any],
                                   yolo_results: Dict[str, Any],
                                   df: pd.DataFrame) -> Dict[str, Any]:
    """Генерация данных для 3D визуализации на основе ИИ анализа (Chronos + YOLO)"""
    
    # Используем health scores из Chronos анализа (более точный)
    health_scores = chronos_results['health_scores']
    anomaly_scores = chronos_results['anomaly_scores']
    
    # Цвета в зависимости от здоровья (Chronos)
    colors = []
    for i, score in enumerate(health_scores):
        # Интенсивность свечения в зависимости от аномалии (YOLO)
        glow = min(anomaly_scores[i] * 2.0, 1.0)
        
        if score > 70:
            # Зеленый - здоров
            colors.append([0, 1, 0, 0.8 + glow * 0.2])
        elif score > 40:
            # Желтый - деградация
            colors.append([1, 1, 0, 0.8 + glow * 0.2])
        else:
            # Красный - критичный
            colors.append([1, 0, 0, 0.8 + glow * 0.2])
    
    # Позиции спутника на орбиталь траектории
    n_points = len(health_scores)
    positions = []
    
    for i in range(n_points):
        # Базовая орбитальная траектория
        angle = (i / n_points) * 2 * np.pi
        radius = 10 + np.sin(angle * 3) * 0.5
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle) * 0.3
        z = radius * np.sin(angle) * 0.7
        
        # Возмущение в зависимости от аномалии (Chronos)
        if anomaly_scores[i] > 0.4:
            perturbation = anomaly_scores[i] * 0.5
            x += np.random.normal(0, perturbation)
            y += np.random.normal(0, perturbation)
            z += np.random.normal(0, perturbation * 0.5)
        
        positions.append([float(x), float(y), float(z)])
    
    # Ключевые моменты из Chronos анализа (слабые сигналы)
    key_moments = []
    
    # Добавляем слабые сигналы из Chronos
    for weak_signal in chronos_results['weak_signals']:
        idx = weak_signal['index']
        key_moments.append({
            "index": idx,
            "time": weak_signal['timestamp'],
            "label": "Weak Signal Detected",
            "description": f"Chronos detected anomaly (score: {weak_signal['anomaly_score']:.2f})",
            "severity": weak_signal['severity'],
            "source": "chronos"
        })
    
    # Добавляем визуальные аномалии из YOLO
    if yolo_results['anomaly_detected']:
        key_moments.append({
            "index": n_points - 1,
            "time": str(df.iloc[-1]['time']),
            "label": "Visual Anomaly Detected",
            "description": f"YOLO detected visual inconsistency (confidence: {yolo_results['confidence']:.2f})",
            "severity": "high" if yolo_results['confidence'] > 0.6 else "medium",
            "source": "yolo"
        })
    
    # Добавляем потерю связи если есть
    if 'comm_status' in df.columns and 'LOST' in df['comm_status'].values:
        lost_idx = df[df['comm_status'] == 'LOST'].index[0]
        key_moments.append({
            "index": int(lost_idx),
            "time": str(df.iloc[lost_idx]['time']),
            "label": "Communication Lost",
            "description": "Complete signal loss detected",
            "severity": "critical",
            "source": "telemetry"
        })
    
    # Сортируем ключевые моменты по времени
    key_moments = sorted(key_moments, key=lambda x: x['index'])
    
    return {
        "positions": positions,
        "colors": colors,
        "key_moments": key_moments,
        "animation_path": positions,
        "health_scores": health_scores,
        "anomaly_scores": anomaly_scores,
        "satellite_scale": [1.0, 1.0, 1.0],
        "light_position": [20, 30, 40],
        "has_data": True,
        "status": "generated",
        "ai_sources": {
            "chronos": {
                "confidence": chronos_results['confidence'],
                "signals_detected": len(chronos_results['weak_signals'])
            },
            "yolo": {
                "confidence": yolo_results['confidence'],
                "anomaly_detected": yolo_results['anomaly_detected']
            }
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": "loaded" if session else "not_loaded"
    }

@app.get("/api/diagnostics")
async def diagnostics():
    """System diagnostics"""
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": "loaded" if session else "not_loaded",
        "model_path": abs_model_path,
        "model_exists": os.path.exists(abs_model_path),
        "analyses_stored": len(analyses_db),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__
    }

@app.post("/api/analyze")
async def analyze_telemetry_file(file: UploadFile = File(...), api_ok: bool = Depends(chronos_api_dep)):
    """Analyze uploaded CSV telemetry file"""
    try:
        print(f"\n📊 Starting file analysis: {file.filename}")
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Файл должен быть в формате CSV"
            )
        
        contents = await file.read()
        print(f"📄 Размер файла: {len(contents)} байт")
        
        try:
            for encoding in ['utf-8', 'cp1251', 'latin1']:
                try:
                    df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                    print(f"✓ Файл прочитан с кодировкой: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8', errors='ignore')))
                print(f"✓ Файл прочитан с игнорированием ошибок кодировки")
        except Exception as e:
            print(f"❌ Ошибка чтения CSV: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка чтения CSV: {str(e)}"
            )
        
        print(f"📋 DataFrame size: {len(df)} rows, {len(df.columns)} columns")
        print(f"📋 Found columns: {list(df.columns)}")
        
        required_columns = ['time', 'battery_temp', 'voltage', 'orientation_error', 'comm_status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Отсутствуют колонки: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Отсутствуют обязательные колонки: {missing_columns}. Найдены: {list(df.columns)}"
            )
        
        # Преобразование типов
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['battery_temp'] = pd.to_numeric(df['battery_temp'], errors='coerce')
        df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')
        df['orientation_error'] = pd.to_numeric(df['orientation_error'], errors='coerce')
        
        # Заполняем пропуски (совместимо с pandas 2.0+)
        df = df.ffill().bfill()
        
        # Удаляем строки со всеми пропусками
        df = df.dropna(how='all')
        
        if len(df) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"После предобработки осталось менее 5 строк данных ({len(df)})"
            )
        
        print(f"✓ Data prepared: {len(df)} rows")
        
        # Анализ телеметрии
        print(f"🔬 Starting telemetry analysis...")
        analysis = analyze_telemetry_data(df)
        print(f"✓ Analysis complete")
        
        # ============= AI-DRIVEN ANALYSIS =============
        # Run Chronos pretrained model for time-series anomaly detection
        print(f"🧠 Running CHRONOS AI model (time-series foundation model)...")
        chronos_results = detect_chronos_anomalies(df)
        print(f"✓ Chronos AI analysis complete - Detected {len(chronos_results['weak_signals'])} weak signals")
        
        # Run YOLO pretrained model for visual anomaly detection
        print(f"👁 Running YOLO AI model (visual anomaly detection)...")
        # Create synthetic frame data from Chronos results
        final_anomaly = chronos_results['anomaly_scores'][-1] if chronos_results['anomaly_scores'] else 0.0
        final_health = 100 * (1 - final_anomaly)
        
        # Map anomaly score to glow intensity (0-1)
        glow_intensity = min(final_anomaly * 2.0, 1.0)
        
        frame_data = {
            'health': final_health,
            'anomaly_score': final_anomaly,
            'timestamp': str(df['time'].iloc[-1]),
            'satellite_color': '#00ff00' if final_health > 70 else '#ffff00' if final_health > 40 else '#ff0000',
            'glow_intensity': glow_intensity
        }
        
        yolo_detector = YOLOAnomalyDetector()
        yolo_results = yolo_detector.analyze_frame(frame_data)
        print(f"✓ YOLO AI analysis complete - Anomaly detected: {yolo_results['anomaly_detected']}")
        
        # Fuse temporal and visual signals
        temporal_anomaly = chronos_results['anomaly_scores'][-1] if chronos_results['anomaly_scores'] else 0.0
        visual_anomaly = yolo_results['confidence']
        fusion_result = yolo_detector.fuse_with_temporal(temporal_anomaly, visual_anomaly)
        print(f"🔀 Signal fusion: {fusion_result['interpretation']}")
        
        # Генерация 3D данных на основе ИИ анализа
        print(f"📐 Generating 3D visualization from AI models (Chronos + YOLO)...")
        visualization_3d = generate_3d_visualization_data(analysis, chronos_results, yolo_results, df)
        print(f"✓ 3D visualization ready (AI-driven)")
        
        # Генерация отчета
        analysis_id = str(uuid.uuid4())
        
        report = {
            "report_id": f"BBX-{analysis_id[:8].upper()}",
            "analysis_id": analysis_id,
            "filename": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "executive_summary": {
                "status": "FAILURE_ANALYZED" if analysis['final_health'] < 30 else "DEGRADATION_DETECTED",
                "root_cause": analysis['root_cause'],
                "confidence": 0.87,
                "criticality": "HIGH" if analysis['final_health'] < 30 else "MEDIUM"
            },
            "timeline_analysis": {
                "degradation_start": analysis['first_anomaly_time'],
                "first_anomaly_detected": analysis['first_anomaly_time'],
                "total_data_points": len(df),
                "time_range": {
                    "start": str(df['time'].iloc[0]),
                    "end": str(df['time'].iloc[-1])
                }
            },
            "health_analysis": {
                "initial_health": analysis['initial_health'],
                "final_health": analysis['final_health'],
                "degradation_rate": f"{analysis['degradation_rate']:.1f}%",
                "trend": "DECLINING" if analysis['final_health'] < analysis['initial_health'] else "STABLE"
            },
            "chronos_analysis": {
                "temporal_anomaly_score": float(temporal_anomaly),
                "weak_signals_detected": len(chronos_results['weak_signals']),
                "regime_shifts": chronos_results['regime_shifts'],
                "confidence": float(chronos_results['confidence']),
                "first_weak_signal": chronos_results['weak_signals'][0] if chronos_results['weak_signals'] else None,
                "health_trend": chronos_results['health_scores']
            },
            "yolo_analysis": {
                "visual_anomaly_score": float(visual_anomaly),
                "anomaly_detected": yolo_results['anomaly_detected'],
                "visual_signals": yolo_results['visual_signals'],
                "detections": yolo_results['detections'],
                "confidence": float(yolo_results['confidence'])
            },
            "signal_fusion": {
                "temporal_score": float(fusion_result['temporal_score']),
                "visual_score": float(fusion_result['visual_score']),
                "fused_score": float(fusion_result['fused_score']),
                "agreement": fusion_result['agreement'],
                "interpretation": fusion_result['interpretation'],
                "confidence": fusion_result['confidence']
            },
            "causal_chain": analysis['causal_chain'],
            "counterfactual_analysis": generate_counterfactual(analysis['root_cause']),
            "recommendations": [
                "Implement early warning system based on Chronos temporal anomalies" if temporal_anomaly > 0.5 else "Current temporal signals are nominal",
                "Monitor visual indicators from 3D system state visualization" if visual_anomaly > 0.5 else "Visual system state is nominal",
                f"Signal agreement level: {fusion_result['agreement']} - {fusion_result['interpretation']}",
                "Increase telemetry report frequency during critical phases",
                "Implement automatic recovery procedures"
            ],
            "visualization_3d": visualization_3d,
            "raw_analysis": analysis
        }
        
        # Сохраняем в память
        analyses_db[analysis_id] = {
            "report": report,
            "dataframe": df.to_dict('records'),
            "analysis": analysis
        }
        
        # Ограничиваем размер памяти
        if len(analyses_db) > 10:
            oldest_key = list(analyses_db.keys())[0]
            del analyses_db[oldest_key]
            print(f"🗑 Удален старый анализ: {oldest_key}")
        
        print(f"✅ Analysis saved: {analysis_id}")
        
        # Convert all numpy types to native Python types
        report_clean = convert_numpy_types(report)
        
        return JSONResponse(
            content={
                "status": "success",
                "analysis_id": analysis_id,
                "report": report_clean,
                "message": f"Анализ завершен. Проанализировано {len(df)} точек данных."
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка анализа: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка анализа: {str(e)}"
        )

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="Анализ не найден")
    
    return {
        "status": "success",
        "analysis": analyses_db[analysis_id]["report"]
    }

@app.get("/api/3d/{analysis_id}")
async def get_3d_visualization_data(analysis_id: str):
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="Анализ не найден")
    
    return {
        "status": "success",
        "visualization_3d": analyses_db[analysis_id]["report"]["visualization_3d"]
    }


@app.api_route('/api/yolo_frame', methods=['GET', 'POST', 'OPTIONS'], dependencies=[Depends(yolo_api_dep)])
async def yolo_frame(request: Request):
    """Accept a base64 data URL image from the frontend and return detections.
    Supports GET for quick connectivity checks and POST for JSON payloads {image, analysis_id}.
    """
    try:
        if request.method == 'GET':
            return { 'status': 'success', 'message': 'YOLO frame endpoint (GET) - send POST with JSON {image, analysis_id, analysis_domain}.' }

        # Handle POST
        payload = await request.json()
        image_data = payload.get('image')
        analysis_id = payload.get('analysis_id')
        analysis_domain = str(payload.get('analysis_domain', 'rocket')).strip().lower()
        conf_threshold = payload.get('conf_threshold', 0.05)
        use_fallback = bool(payload.get('use_fallback', False))

        if not image_data:
            return JSONResponse(status_code=400, content={'status': 'error', 'message': 'No image provided'})

        # Decode data URL if present
        import base64, re
        m = re.match(r'data:(image/\w+);base64,(.*)', image_data)
        saved_path = None
        if m:
            img_type = m.group(1).split('/')[-1]
            img_b64 = m.group(2)
            img_bytes = base64.b64decode(img_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + img_type) as f:
                f.write(img_bytes)
                saved_path = f.name
        else:
            # If not a data URL, try to treat as raw base64
            try:
                img_bytes = base64.b64decode(image_data)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                    f.write(img_bytes)
                    saved_path = f.name
            except Exception:
                saved_path = None

        # Run YOLO detection with domain-specific checkpoints.
        def resolve_domain_model_path(domain: str) -> Optional[Path]:
            backend_dir = Path(__file__).resolve().parent
            app_root = backend_dir.parent
            workspace_root = app_root.parent
            if domain == "spacesuit":
                candidates = [
                    app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train4_obb_retrain" / "weights" / "best.pt",
                    app_root / "spacesuit damage detection.v1i.yolov8-obb" / "runs" / "detect" / "train3" / "weights" / "best.pt",
                    app_root / "spacesuit" / "train3" / "weights" / "best.pt",
                    workspace_root / "spacesuit" / "train3" / "weights" / "best.pt",
                ]
            else:
                candidates = [
                    app_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
                    app_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "last.pt",
                    workspace_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
                    workspace_root / "rocket yolo" / "runs" / "detect" / "train" / "weights" / "last.pt",
                ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
            return None

        model_path = resolve_domain_model_path(analysis_domain)
        detections = []
        model_info = {}
        detection_mode = "none"
        try:
            detector = YOLOAnomalyDetector(model_path=str(model_path) if model_path else None)
            model_info = detector.get_model_info() if hasattr(detector, 'get_model_info') else {}
            if saved_path and hasattr(detector, 'analyze_image_path'):
                detections = detector.analyze_image_path(
                    saved_path,
                    conf_threshold=float(conf_threshold),
                    use_fallback=use_fallback
                )
                if detections:
                    detection_mode = detections[0].get('source', 'yolo')
        except Exception as detector_error:
            model_info = {'model_loaded': False, 'load_error': str(detector_error)}
            detections = []

        # Cleanup saved file optionally
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass

        return JSONResponse(content=convert_numpy_types({
            'status': 'success',
            'message': 'YOLO frame received',
            'analysis_id': analysis_id,
            'analysis_domain': analysis_domain,
            'detections': detections,
            'detection_count': len(detections),
            'anomaly_detected': len(detections) > 0,
            'detection_mode': detection_mode,
            'model': model_info
        }))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post("/analyze/voice")
async def analyze_voice(
    audio_file: UploadFile = File(...),
    baseline_file: Optional[UploadFile] = File(None),
    audio_type: str = Form("generic"),
    report_mode: str = Form("full"),
):
    """
    Multipart voice diagnostics endpoint.
    Fields:
      - audio_file: required
      - baseline_file: optional
      - audio_type: optional
      - report_mode: optional
    """
    try:
        if not audio_file:
            raise HTTPException(status_code=400, detail="audio_file is required")

        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="audio_file is empty")

        baseline_bytes = None
        if baseline_file is not None:
            baseline_bytes = await baseline_file.read()
            if baseline_bytes is not None and len(baseline_bytes) == 0:
                baseline_bytes = None

        audio_suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        baseline_suffix = ".wav"
        if baseline_file and baseline_file.filename:
            baseline_suffix = Path(baseline_file.filename).suffix or ".wav"

        result = analyze_voice_bytes(
            audio_bytes=audio_bytes,
            audio_suffix=audio_suffix,
            baseline_bytes=baseline_bytes,
            baseline_suffix=baseline_suffix,
            config=VoiceAnalyzeConfig(audio_type=audio_type, report_mode=report_mode),
        )
        return JSONResponse(content=convert_numpy_types(result.model_dump()))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice analysis error: {str(e)}")

@app.post("/api/agent/query")
async def agent_query(query: AgentQuery):
    
    if query.analysis_id not in analyses_db:
        return {
            "answer": "Анализ не найден. Пожалуйста, загрузите телеметрию для анализа.",
            "confidence": 0.0,
            "sources": []
        }
    
    analysis = analyses_db[query.analysis_id]["report"]
    
    question_lower = query.question.lower()
    
    if any(word in question_lower for word in ['корень', 'причина', 'почему', 'root', 'cause']):
        answer = f"Коренная причина: {analysis['executive_summary']['root_cause']}. "
        answer += f"Уверенность анализа: {analysis['executive_summary']['confidence']*100:.1f}%."
        sources = ["root_cause_analysis", "pattern_detection"]
        
    elif any(word in question_lower for word in ['когда', 'начало', 'start', 'time']):
        if analysis['timeline_analysis']['degradation_start'] != "N/A":
            answer = f"Деградация началась в {analysis['timeline_analysis']['degradation_start']}. "
        else:
            answer = "Четкий момент начала деградации не обнаружен. "
        answer += f"Анализ охватывает период с {analysis['timeline_analysis']['time_range']['start']} по {analysis['timeline_analysis']['time_range']['end']}."
        sources = ["timeline_analysis", "anomaly_detection"]
        
    elif any(word in question_lower for word in ['здоровье', 'health', 'статус', 'status']):
        health = analysis['health_analysis']
        answer = f"Начальное здоровье: {health['initial_health']:.1f}%. "
        answer += f"Финальное здоровье: {health['final_health']:.1f}%. "
        answer += f"Тренд: {health['trend']}. "
        answer += f"Скорость деградации: {health['degradation_rate']}."
        sources = ["health_metrics", "trend_analysis"]
        
    elif any(word in question_lower for word in ['цепочка', 'chain', 'последовательность']):
        chain = analysis['causal_chain']
        answer = "Цепочка отказа: " + " → ".join(chain) + "."
        sources = ["causal_analysis", "failure_patterns"]
        
    elif any(word in question_lower for word in ['предотвратить', 'избежать', 'prevent', 'avoid']):
        answer = analysis['counterfactual_analysis']
        sources = ["counterfactual_analysis", "expert_rules"]
        
    elif any(word in question_lower for word in ['рекомендации', 'recommendations', 'совет']):
        recs = analysis['recommendations']
        answer = "Рекомендации: " + "; ".join(recs) + "."
        sources = ["recommendation_engine", "best_practices"]
        
    else:
        answer = f"На основе анализа телеметрии, спутник показал {analysis['health_analysis']['trend']} тренд здоровья. "
        answer += f"Основная проблема: {analysis['executive_summary']['root_cause']}. "
        answer += "Задайте более конкретный вопрос о причине, времени или рекомендациях."
        sources = ["general_analysis"]
    
    return {
        "answer": answer,
        "confidence": 0.85,
        "sources": sources,
        "analysis_id": query.analysis_id
    }

@app.get("/api/sample-csv")
async def download_sample_csv():
    """Скачать образец CSV файла для тестирования"""
    
    sample_data = []
    start_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    # Нормальная работа
    for i in range(50):
        time = start_time + pd.Timedelta(minutes=i)
        sample_data.append({
            'time': time.isoformat(),
            'battery_temp': 25 + np.random.normal(0, 1),
            'voltage': 28 + np.random.normal(0, 0.5),
            'orientation_error': 1.0 + np.random.exponential(0.5),
            'comm_status': 'OK'
        })
    
    # Деградация
    for i in range(30):
        time = start_time + pd.Timedelta(minutes=50 + i)
        degradation = i / 30
        
        sample_data.append({
            'time': time.isoformat(),
            'battery_temp': 25 + degradation * 15 + np.random.normal(0, 2),
            'voltage': 28 - degradation * 8 + np.random.normal(0, 1),
            'orientation_error': 1.0 + degradation * 10 + np.random.exponential(1),
            'comm_status': 'DEGRADED' if i < 20 else 'LOST'
        })
    
    df = pd.DataFrame(sample_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='text/csv',
        filename='sample_telemetry.csv'
    )


# Mount frontend at root last so API routes take precedence (styles, scripts, all pages work)
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print(f"✓ Frontend mounted at root: {frontend_path}")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting NOEMA server...")
    print("📊 Open in browser: http://localhost:8000")
    print("📋 API docs: http://localhost:8000/api/docs")
    # Print API key enforcement status
    print('\nAPI key enforcement settings:')
    print(f"  CHRONOS_API_KEY set: {bool(CHRONOS_API_KEY)}")
    print(f"  YOLO_API_KEY set: {bool(YOLO_API_KEY)}")
    print(f"  ADMIN_API_KEY set: {bool(ADMIN_API_KEY)}")
    # Print registered routes for debugging
    try:
        print('\nRegistered routes:')
        for route in app.routes:
            methods = getattr(route, 'methods', None)
            path = getattr(route, 'path', getattr(route, 'url', str(route)))
            print(f"  {path} -> {methods}")
    except Exception as e:
        print('Could not list routes:', e)
    uvicorn.run(app, host="0.0.0.0", port=8000)
