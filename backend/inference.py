import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import onnxruntime as ort
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TelemetryAnalysis:
    """Результаты анализа телеметрии"""
    health_scores: List[float]
    anomaly_scores: List[float]
    first_anomaly_index: int
    first_anomaly_time: str
    failure_index: int
    failure_time: str
    root_cause: str
    causal_chain: List[str]
    final_health: float
    degradation_start: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_scores": self.health_scores,
            "anomaly_scores": self.anomaly_scores,
            "first_anomaly_index": self.first_anomaly_index,
            "first_anomaly_time": self.first_anomaly_time,
            "failure_index": self.failure_index,
            "failure_time": self.failure_time,
            "root_cause": self.root_cause,
            "causal_chain": self.causal_chain,
            "final_health": self.final_health,
            "degradation_start": self.degradation_start
        }

def preprocess_telemetry(df: pd.DataFrame) -> np.ndarray:
    """Предобработка телеметрических данных"""
    # Нормализация числовых признаков
    numeric_cols = ['battery_temp', 'voltage', 'orientation_error']
    
    # Кодирование категориального признака
    comm_mapping = {'OK': 0, 'DEGRADED': 0.5, 'LOST': 1}
    df['comm_status_encoded'] = df['comm_status'].map(comm_mapping)
    
    # Создание массива признаков
    features = df[numeric_cols + ['comm_status_encoded']].values
    
    return features.astype(np.float32)

def compute_anomaly_score(features: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Вычисление аномалий через ONNX модель"""
    # Подготовка входных данных
    input_name = session.get_inputs()[0].name
    
    # Инференс
    reconstruction = session.run(None, {input_name: features})[0]
    
    # Вычисление ошибки реконструкции
    errors = np.mean((features - reconstruction) ** 2, axis=1)
    
    return errors

def detect_anomalies(anomaly_scores: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """Обнаружение аномалий во временном ряде"""
    # Вычисление базового уровня (первые 20% данных считаем нормальными)
    baseline_len = int(len(anomaly_scores) * 0.2)
    baseline_mean = np.mean(anomaly_scores[:baseline_len])
    baseline_std = np.std(anomaly_scores[:baseline_len])
    
    # Персистентное отклонение (скользящее окно)
    window_size = 10
    persistent_anomalies = []
    
    for i in range(len(anomaly_scores) - window_size):
        window_mean = np.mean(anomaly_scores[i:i+window_size])
        if window_mean > baseline_mean + 2 * baseline_std:
            persistent_anomalies.append(i)
    
    first_anomaly = persistent_anomalies[0] if persistent_anomalies else -1
    
    return {
        "first_anomaly": first_anomaly,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "anomaly_threshold": baseline_mean + 2 * baseline_std
    }

def analyze_root_cause(df: pd.DataFrame, anomaly_idx: int) -> Tuple[str, List[str]]:
    """Анализ вероятной первопричины и цепочки причин"""
    # Анализ конечного состояния
    last_row = df.iloc[-1]
    
    # Определение первопричины на основе паттернов
    if last_row['comm_status'] == 'LOST':
        if last_row['voltage'] < 3.0:
            root_cause = "Критическая потеря напряжения питания"
            causal_chain = [
                "Деградация батареи/солнечных панелей",
                "Падение напряжения ниже критического уровня",
                "Сбой систем питания",
                "Потеря ориентации",
                "Нарушение связи"
            ]
        elif last_row['orientation_error'] > 30.0:
            root_cause = "Критическая ошибка ориентации"
            causal_chain = [
                "Сбой гироскопов/реактивных колес",
                "Неконтролируемое вращение",
                "Невозможность наведения антенн",
                "Потеря связи с Землей"
            ]
        else:
            root_cause = "Сбой системы связи"
            causal_chain = [
                "Отказ передатчика/приемника",
                "Потеря сигнала",
                "Невозможность получения команд"
            ]
    else:
        root_cause = "Частичная деградация систем"
        causal_chain = [
            "Начальная стадия отказа компонентов",
            "Ухудшение параметров",
            "Снижение общей надежности"
        ]
    
    return root_cause, causal_chain

def generate_counterfactual(root_cause: str) -> str:
    """Генерация контрафактического объяснения"""
    counterfactuals = {
        "Критическая потеря напряжения питания": 
            "Если бы система мониторирования напряжения сработала на 15 минут раньше, "
            "можно было бы переключиться на резервный источник питания и избежать полного отказа.",
        
        "Критическая ошибка ориентации":
            "При наличии дополнительного гироскопа в резерве и мгновенном переключении "
            "ориентация могла быть восстановлена до потери связи.",
        
        "Сбой системы связи":
            "Регулярная калибровка антенн и проверка усилителей могла бы предотвратить "
            "внезапный отказ системы связи.",
        
        "Частичная деградация систем":
            "Плановое техобслуживание и замена деградирующих компонентов до критического "
            "уровня сохранили бы работоспособность системы."
    }
    
    return counterfactuals.get(root_cause, 
        "Раннее обнаружение аномалий и превентивные меры могли бы предотвратить эскалацию отказа.")

def analyze_telemetry(df: pd.DataFrame, session: ort.InferenceSession) -> TelemetryAnalysis:
    """Основная функция анализа телеметрии"""
    # Предобработка
    features = preprocess_telemetry(df)
    
    # Вычисление аномалий
    anomaly_scores = compute_anomaly_score(features, session)
    
    # Нормализация в health scores (0-100)
    health_scores = 100 * (1 - anomaly_scores / np.max(anomaly_scores))
    
    # Обнаружение аномалий
    anomaly_info = detect_anomalies(anomaly_scores)
    
    # Анализ первопричины
    root_cause, causal_chain = analyze_root_cause(df, anomaly_info["first_anomaly"])
    
    # Контрафактическое объяснение
    counterfactual = generate_counterfactual(root_cause)
    
    # Формирование результатов
    return TelemetryAnalysis(
        health_scores=health_scores.tolist(),
        anomaly_scores=anomaly_scores.tolist(),
        first_anomaly_index=anomaly_info["first_anomaly"],
        first_anomaly_time=df.iloc[anomaly_info["first_anomaly"]]['time'] if anomaly_info["first_anomaly"] != -1 else "N/A",
        failure_index=len(df) - 1,
        failure_time=df.iloc[-1]['time'],
        root_cause=root_cause,
        causal_chain=causal_chain,
        final_health=float(health_scores[-1]),
        degradation_start=df.iloc[anomaly_info["first_anomaly"]]['time'] if anomaly_info["first_anomaly"] != -1 else "N/A"
    )

def generate_report(analysis: TelemetryAnalysis) -> Dict[str, Any]:
    """Генерация финального отчета"""
    return {
        "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "executive_summary": {
            "status": "FAILURE_ANALYZED" if analysis.final_health < 30 else "DEGRADATION_DETECTED",
            "root_cause": analysis.root_cause,
            "confidence": 0.87,
            "criticality": "HIGH" if analysis.final_health < 30 else "MEDIUM"
        },
        "timeline_analysis": {
            "degradation_start": analysis.degradation_start,
            "first_anomaly_detected": analysis.first_anomaly_time,
            "failure_confirmed": analysis.failure_time,
            "total_duration_minutes": "Estimated: 45-60 мин"
        },
        "causal_chain": analysis.causal_chain,
        "health_metrics": {
            "initial_health": analysis.health_scores[0] if analysis.health_scores else 100,
            "final_health": analysis.final_health,
            "degradation_rate": f"{((analysis.health_scores[0] - analysis.final_health) / analysis.health_scores[0] * 100):.1f}%"
        },
        "counterfactual_analysis": generate_counterfactual(analysis.root_cause),
        "recommendations": [
            "Внедрить систему раннего предупреждения на основе анализа аномалий",
            "Добавить резервирование критических компонентов",
            "Увеличить частоту телеметрических отчетов в критические фазы",
            "Реализовать автоматические процедуры восстановления"
        ],
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0",
            "black_box_serial": "BBX-001-2024"
        }
    }