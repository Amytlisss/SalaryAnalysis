import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
import math
import base64
from io import BytesIO

warnings.filterwarnings('ignore')


class SalaryAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.warnings = []
        self._normalize_column_names()
        self._validate_data()

    def _normalize_column_names(self):
        column_mapping = {
            'стаж': 'Стаж',
            'experience': 'Стаж',
            'exp': 'Стаж',
            'Experience': 'Стаж',
            'EXP': 'Стаж',
            'Стаж (годы)': 'Стаж',
            'Стаж работы': 'Стаж',
            'зарплата': 'Зарплата',
            'salary': 'Зарплата',
            'Salary': 'Зарплата',
            'ЗП': 'Зарплата',
            'зп': 'Зарплата',
            'Заработная плата': 'Зарплата',
            'Заработная плата, руб.': 'Зарплата',
            'Оклад': 'Зарплата'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns and old_name != new_name:
                self.df = self.df.rename(columns={old_name: new_name})
                self.warnings.append(f"Колонка '{old_name}' переименована в '{new_name}'")

    def _validate_data(self):
        required_columns = ['Стаж', 'Зарплата']

        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Отсутствует обязательный столбец: {col}")

        for col in required_columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                self.warnings.append(
                    f"Столбец '{col}' содержит {missing} пропущенных значений. Они будут исключены."
                )
                self.df = self.df.dropna(subset=[col])

        self.df['Стаж'] = pd.to_numeric(self.df['Стаж'], errors='coerce')
        self.df['Зарплата'] = pd.to_numeric(self.df['Зарплата'], errors='coerce')
        self.df = self.df.dropna(subset=required_columns)

        if (self.df['Стаж'] < 0).any():
            negative_exp = (self.df['Стаж'] < 0).sum()
            self.warnings.append(
                f"Обнаружено {negative_exp} записей с отрицательным стажем. Они будут исключены."
            )
            self.df = self.df[self.df['Стаж'] >= 0]

        if (self.df['Зарплата'] <= 0).any():
            non_pos_sal = (self.df['Зарплата'] <= 0).sum()
            self.warnings.append(
                f"Обнаружено {non_pos_sal} записей с нулевой или отрицательной зарплатой. Они будут исключены."
            )
            self.df = self.df[self.df['Зарплата'] > 0]

        for col in required_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                self.warnings.append(f"Столбец '{col}' содержит {len(outliers)} потенциальных выбросов")

        if len(self.df) < 2:
            raise ValueError(
                f"Недостаточно данных для анализа: осталось {len(self.df)} записей. Нужно минимум 2."
            )

    def calculate_correlation(self):
        x = self.df['Стаж'].values.astype(float)
        y = self.df['Зарплата'].values.astype(float)

        pearson_coef, pearson_p = stats.pearsonr(x, y)
        spearman_coef, spearman_p = stats.spearmanr(x, y)

        abs_pearson = abs(pearson_coef)
        if abs_pearson < 0.3:
            interpretation = "слабая корреляционная связь"
        elif abs_pearson < 0.7:
            interpretation = "умеренная корреляционная связь"
        else:
            interpretation = "сильная корреляционная связь"

        direction = "прямая" if pearson_coef > 0 else "обратная"

        return float(pearson_coef), float(spearman_coef), f"{direction}, {interpretation}"

    def calculate_partial_correlation(self, control_cols=None):
        if control_cols is None:
            control_cols = ['Должность', 'Образование', 'Отдел']
        
        available_controls = [col for col in control_cols if col in self.df.columns]
        
        if not available_controls:
            self.warnings.append("Не найдены столбцы для расчета частной корреляции")
            return None
        
        # Создаём копию данных
        df_encoded = self.df[['Стаж', 'Зарплата'] + available_controls].copy()
        df_encoded = df_encoded.dropna()
        
        if len(df_encoded) < 10:
            self.warnings.append(
                f"Недостаточно данных для частной корреляции: {len(df_encoded)} записей, нужно минимум 10"
            )
            return None
        
        for col in available_controls:
            if col in df_encoded.columns:
                df_encoded[col], _ = pd.factorize(df_encoded[col])
        
        # Преобразуем в числовой тип
        for col in available_controls:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
        
        df_encoded = df_encoded.dropna()
        
        if len(df_encoded) < 10:
            self.warnings.append(
                f"После преобразования осталось {len(df_encoded)} записей, нужно минимум 10"
            )
            return None
        
        # Проверяем уникальность значений
        unique_counts = {col: df_encoded[col].nunique() for col in available_controls}
        for col, count in unique_counts.items():
            if count < 2:
                self.warnings.append(
                    f"Столбец '{col}' имеет только {count} уникальных значений, частная корреляция может быть неточной"
                )
        
        X_control = df_encoded[available_controls].values.astype(float)
        X_control = sm.add_constant(X_control)
        
        y = df_encoded['Зарплата'].values.astype(float)
        x = df_encoded['Стаж'].values.astype(float)
        
        try:
            model_y = sm.OLS(y, X_control).fit()
            residuals_y = model_y.resid
            
            model_x = sm.OLS(x, X_control).fit()
            residuals_x = model_x.resid
            
            partial_corr, p_value = stats.pearsonr(residuals_x, residuals_y)
            
            return {
                'coefficient': float(partial_corr),
                'p_value': float(p_value),
                'controlled_factors': available_controls
            }
        except Exception as e:
            self.warnings.append(f"Не удалось рассчитать частную корреляцию: {str(e)}")
            return None

    def calculate_regression(self, degree=1):
        X = self.df['Стаж'].values.astype(float)
        y = self.df['Зарплата'].values.astype(float)

        if degree == 2:
            X_poly = np.column_stack([X, X**2])
            X_poly = sm.add_constant(X_poly)
            model = sm.OLS(y, X_poly).fit()

            equation = (
                f"Зарплата = {model.params[0]:.2f} + "
                f"{model.params[1]:.2f}·Стаж + {model.params[2]:.2f}·Стаж²"
            )

            coefficients = {
                'const': float(model.params[0]),
                'Стаж': float(model.params[1]),
                'Стаж²': float(model.params[2])
            }
            p_values = {
                'const': float(model.pvalues[0]),
                'Стаж': float(model.pvalues[1]),
                'Стаж²': float(model.pvalues[2])
            }
        else:
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()

            equation = f"Зарплата = {model.params[0]:.2f} + {model.params[1]:.2f}·Стаж"

            coefficients = {
                'const': float(model.params[0]),
                'Стаж': float(model.params[1])
            }
            p_values = {
                'const': float(model.pvalues[0]),
                'Стаж': float(model.pvalues[1])
            }

        return {
            'equation': equation,
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'std_error': float(np.sqrt(model.mse_resid)),
            'coefficients': coefficients,
            'p_values': p_values,
            'model': model
        }

    def generate_plots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        plots = {}

        X = self.df['Стаж'].values.astype(float)
        y = self.df['Зарплата'].values.astype(float)

        # 1. Диаграмма рассеяния с линией регрессии
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model = self.calculate_regression(degree=1)['model']
        X_sorted = np.sort(X)
        X_with_const = sm.add_constant(X_sorted)
        y_pred = model.predict(X_with_const)

        ax.scatter(X, y, alpha=0.6, color='steelblue', label='Данные')
        ax.plot(X_sorted, y_pred, color='coral', linewidth=2, label='Линия регрессии')

        ax.set_xlabel('Стаж (лет)', fontsize=12)
        ax.set_ylabel('Зарплата (руб.)', fontsize=12)
        ax.set_title('Зависимость зарплаты от стажа работы', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['scatter'] = base64.b64encode(buffer.read()).decode()
        plt.close()

        # 2. Матрица корреляций (с кодированием категориальных переменных)

        fig, ax = plt.subplots(figsize=(10, 8))

        df_for_corr = self.df.copy()

        # Кодируем только категориальные переменные
        categorical_cols = ['Должность', 'Образование', 'Отдел']
        for col in categorical_cols:
            if col in df_for_corr.columns:
                df_for_corr[col] = pd.Categorical(df_for_corr[col]).codes

        #Выбираем только числовые колонки (включая закодированные категориальные)
        numeric_cols = df_for_corr.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            corr_matrix = df_for_corr[numeric_cols].corr()
            
            # Показываем нижний треугольник + диагональ
            mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
            mask = ~mask
            
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1
            )
            ax.set_title('Матрица корреляций', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plots['correlation_heatmap'] = base64.b64encode(buffer.read()).decode()
        plt.close()
        # 3. Гистограмма распределения зарплат (Правило Стёрджесса)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_val = np.mean(y)
        median_val = np.median(y)
        
        n = len(y)
        k = math.ceil(math.log2(n) + 1)

        ax.hist(y, bins=k, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {mean_val:.0f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Медиана: {median_val:.0f}')

        ax.set_xlabel('Зарплата (руб.)', fontsize=12)
        ax.set_ylabel('Количество сотрудников', fontsize=12)
        ax.set_title('Распределение заработной платы', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plots['salary_histogram'] = base64.b64encode(buffer.read()).decode()
        plt.close()

        return plots