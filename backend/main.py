from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import pandas as pd
import numpy as np
import io
import math
import os
from typing import Optional
from analysis import SalaryAnalyzer
from models import AnalysisResponse, CorrelationResult, RegressionResult, PartialCorrelationResult

app = FastAPI(title="SalaryAnalysis API", description="API для корреляционно-регрессионного анализа кадровых данных")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_for_json(obj):
    """Рекурсивно заменяет NaN, Infinity и None на значения, безопасные для JSON"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # Заменяем NaN на 0 для безопасной сериализации
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif obj is None:
        return None
    return obj


@app.get("/")
async def root():
    return {"message": "SalaryAnalysis API", "status": "running"}


@app.post("/api/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    calculate_partial: bool = True,
    regression_degree: int = 1
):
    """
    Анализ загруженных данных:
    - Расчёт корреляций Пирсона и Спирмена
    - Частная корреляция (опционально)
    - Регрессионная модель (линейная или полиномиальная)
    - Генерация графиков
    """
    # Проверка расширения файла
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Поддерживаются только файлы CSV и XLSX")

    try:
        # Чтение файла
        contents = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Анализ данных (SalaryAnalyzer сам нормализует колонки)
        analyzer = SalaryAnalyzer(df)
        
        # Расчёт корреляций
        pearson, spearman, interpretation = analyzer.calculate_correlation()
        
        # Частная корреляция
        partial_result = None
        if calculate_partial:
            partial = analyzer.calculate_partial_correlation()
            if partial:
                partial_result = PartialCorrelationResult(
                    coefficient=partial['coefficient'],
                    p_value=partial['p_value'],
                    controlled_factors=partial['controlled_factors']
                )
        
        # Регрессионная модель
        regression_data = analyzer.calculate_regression(degree=regression_degree)
        regression = RegressionResult(
            equation=regression_data['equation'],
            r_squared=regression_data['r_squared'],
            adj_r_squared=regression_data['adj_r_squared'],
            std_error=regression_data['std_error'],
            coefficients=regression_data['coefficients'], 
            p_values=regression_data['p_values']
        )
        
        # Генерация графиков
        plots = analyzer.generate_plots()
        
        # Формирование ответа
        response = {
            "correlation": {
                "pearson": pearson,
                "spearman": spearman,
                "interpretation": interpretation
            },
            "partial_correlation": partial_result.dict() if partial_result else None,
            "regression": regression.dict(),
            "sample_size": len(analyzer.df),
            "warnings": analyzer.warnings,
            "plots": plots
        }
        
        # Очищаем ответ от NaN перед отправкой
        return JSONResponse(content=clean_for_json(response))
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Ошибка валидации данных: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе данных: {str(e)}")


@app.post("/api/export-report")
async def export_report(file: UploadFile = File(...)):
    """
    Экспорт отчёта (вспомогательный эндпоинт)
    """
    try:
        contents = await file.read()
        filename = file.filename
        return StreamingResponse(
            io.BytesIO(contents),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=analyzed_{filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте: {str(e)}")


# Путь к фронтенду
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

@app.get("/ui")
async def serve_ui():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/{filename}")
async def serve_static(filename: str):
    file_path = os.path.join(frontend_path, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)