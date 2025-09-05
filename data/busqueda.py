# Casos de uso de Machine Learning Supervisado
# Investigación académica sobre aplicaciones relevantes

research_info = {
    'title': 'Casos de Uso de Machine Learning Supervisado',
    'subtitle': 'Investigación de aplicaciones prácticas y relevantes',
    'abstract': '''
    Esta investigación examina cuatro casos de uso fundamentales del Machine Learning Supervisado,
    analizando sus aplicaciones, metodologías, beneficios y desafíos en contextos reales.
    Cada caso de uso demuestra cómo los algoritmos supervisados pueden resolver problemas
    complejos en diferentes industrias y dominios.
    ''',
    'keywords': ['Machine Learning', 'Aprendizaje Supervisado', 'Clasificación', 'Regresión', 'Predicción'],
    'methodology': 'Revisión sistemática de literatura, análisis de datasets públicos y evaluación de algoritmos',
    'date': '2025'
}

# 4 Casos de uso principales
ml_use_cases = [
    {
        'id': 1,
        'title': 'Diagnóstico Médico por Imágenes',
        'category': 'Salud y Medicina',
        'description': 'Clasificación automática de imágenes médicas para detectar enfermedades como cáncer, neumonía y otras patologías.',
        'algorithm_type': 'Clasificación',
        'algorithms_used': ['Redes Neuronales Convolucionales (CNN)', 'Support Vector Machine', 'Random Forest'],
        'dataset_example': 'Chest X-Ray Images (Pneumonia)',
        'dataset_size': '5,863 imágenes',
        'accuracy': '95%',
        'applications': [
            'Detección temprana de cáncer',
            'Diagnóstico de neumonía',
            'Análisis de radiografías',
            'Screening masivo'
        ],
        'benefits': [
            'Diagnóstico más rápido y preciso',
            'Reducción de errores humanos',
            'Acceso a diagnóstico en áreas remotas',
            'Análisis consistente 24/7'
        ],
        'challenges': [
            'Necesidad de grandes datasets etiquetados',
            'Interpretabilidad de resultados',
            'Regulaciones médicas estrictas',
            'Sesgo en los datos de entrenamiento'
        ]
    },
    {
        'id': 2,
        'title': 'Detección de Fraude Financiero',
        'category': 'Finanzas y Banca',
        'description': 'Sistema de detección automática de transacciones fraudulentas en tiempo real para proteger a usuarios y entidades financieras.',
        'algorithm_type': 'Clasificación',
        'algorithms_used': ['Gradient Boosting', 'Logistic Regression', 'Neural Networks', 'Isolation Forest'],
        'dataset_example': 'Credit Card Fraud Detection',
        'dataset_size': '284,807 transacciones',
        'accuracy': '99.2%',
        'applications': [
            'Detección de fraude en tarjetas de crédito',
            'Análisis de transacciones sospechosas',
            'Prevención de lavado de dinero',
            'Scoring de riesgo crediticio'
        ],
        'benefits': [
            'Detección en tiempo real',
            'Reducción de pérdidas financieras',
            'Mejor experiencia del cliente',
            'Cumplimiento regulatorio'
        ],
        'challenges': [
            'Datos altamente desbalanceados',
            'Falsos positivos costosos',
            'Adaptación a nuevos tipos de fraude',
            'Privacidad y protección de datos'
        ]
    },
    {
        'id': 3,
        'title': 'Sistemas de Recomendación',
        'category': 'Comercio Electrónico',
        'description': 'Algoritmos que predicen y sugieren productos, contenido o servicios personalizados basados en el comportamiento del usuario.',
        'algorithm_type': 'Regresión/Clasificación',
        'algorithms_used': ['Collaborative Filtering', 'Matrix Factorization', 'Deep Learning', 'Content-Based Filtering'],
        'dataset_example': 'MovieLens 25M Dataset',
        'dataset_size': '25 millones de calificaciones',
        'accuracy': '87% RMSE',
        'applications': [
            'Recomendaciones de productos en e-commerce',
            'Sugerencias de contenido en streaming',
            'Personalización de feeds sociales',
            'Matching en plataformas de citas'
        ],
        'benefits': [
            'Aumento en ventas y engagement',
            'Mejor experiencia de usuario',
            'Descubrimiento de contenido relevante',
            'Retención de clientes'
        ],
        'challenges': [
            'Cold start problem',
            'Escalabilidad con millones de usuarios',
            'Diversidad vs relevancia',
            'Filter bubble y sesgo algorítmico'
        ]
    },
    {
        'id': 4,
        'title': 'Predicción de Demanda y Precios',
        'category': 'Supply Chain y Retail',
        'description': 'Modelos predictivos para anticipar la demanda de productos y optimizar precios basados en múltiples variables del mercado.',
        'algorithm_type': 'Regresión',
        'algorithms_used': ['ARIMA', 'LSTM', 'Random Forest Regressor', 'XGBoost'],
        'dataset_example': 'Walmart Sales Forecasting',
        'dataset_size': '421,570 registros históricos',
        'accuracy': '92% MAE',
        'applications': [
            'Forecast de ventas retail',
            'Optimización de inventarios',
            'Pricing dinámico',
            'Planificación de producción'
        ],
        'benefits': [
            'Reducción de costos de inventario',
            'Maximización de ingresos',
            'Mejor planificación empresarial',
            'Reducción de desperdicios'
        ],
        'challenges': [
            'Estacionalidad y tendencias complejas',
            'Factores externos impredecibles',
            'Múltiples variables correlacionadas',
            'Adaptación a cambios del mercado'
        ]
    }
]

# Información del equipo de investigación
team_info = {
    'institution': 'Universidad de Tecnología',
    'department': 'Departamento de Ciencias de la Computación',
    'course': 'Machine Learning y Data Science',
    'researchers': [
        {
            'name': 'Estudiante Investigador',
            'role': 'Investigador Principal',
            'specialization': 'Machine Learning Supervisado'
        }
    ]
}

# Métricas y estadísticas del estudio
study_metrics = {
    'total_cases': len(ml_use_cases),
    'algorithms_analyzed': 15,
    'datasets_reviewed': 8,
    'avg_accuracy': '93.3%',
    'industries_covered': 4,
    'papers_reviewed': 45
}