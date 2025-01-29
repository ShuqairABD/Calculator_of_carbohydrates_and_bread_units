import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1
from torchvision import transforms
from PIL import Image
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error


# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# Классы еды
class_names = ['falafel', 'hamburger', 'hummus', 'lasagna', 'paella', 'pizza', 'poutine', 'ramen', 'ravioli', 'samosa']
carbohydrates_per_100g = {
    "falafel": 71.60, "hamburger": 30.25, "hummus": 8.50, "lasagna": 18.70, "paella": 21.20,
    "pizza": 35, "poutine": 71.50, "ramen": 30.10, "ravioli": 14.67, "samosa": 24.20
}

# Напитки
drinks_data = {
    "Кока-Кола": {"carbs_per_100ml": 10.6, "HE_per_100ml": 10.6 / 12},
    "Фанта": {"carbs_per_100ml": 12, "HE_per_100ml": 12 / 12},
    "Спрайт": {"carbs_per_100ml": 9, "HE_per_100ml": 9 / 12},
    "Квас": {"carbs_per_100ml": 6, "HE_per_100ml": 6 / 12},
    "Сок яблочный": {"carbs_per_100ml": 11.5, "HE_per_100ml": 11.5 / 12},
    "Сок апельсиновый": {"carbs_per_100ml": 12, "HE_per_100ml": 12 / 12},
    "Молоко": {"carbs_per_100ml": 4.7, "HE_per_100ml": 4.7 / 12},
    "Айран": {"carbs_per_100ml": 2, "HE_per_100ml": 2 / 12},
    "Чай с сахаром": {"carbs_per_100ml": 0, "HE_per_100ml": 0},  # сахар отдельно
    "Кофе с сахаром": {"carbs_per_100ml": 0, "HE_per_100ml": 0},  # сахар отдельно
    "Дюшес": {"carbs_per_100ml": 9.7, "HE_per_100ml": 9.7 / 12},
    "Тархун": {"carbs_per_100ml": 11.5, "HE_per_100ml": 11.5 / 12},
    "Вода": {"carbs_per_100ml": 0, "HE_per_100ml": 0},
    "Чай без сахара": {"carbs_per_100ml": 0, "HE_per_100ml": 0},
    "Кофе без сахара": {"carbs_per_100ml": 0, "HE_per_100ml": 0},
    "Минеральная вода": {"carbs_per_100ml": 0, "HE_per_100ml": 0},
    "Пепси": {"carbs_per_100ml": 11, "HE_per_100ml": 11 / 12},
    "Лимонад": {"carbs_per_100ml": 9, "HE_per_100ml": 9 / 12},
    "Морс": {"carbs_per_100ml": 14, "HE_per_100ml": 14 / 12},
    "Какао на молоке": {"carbs_per_100ml": 10.50, "HE_per_100ml": 10.50 / 12},
    "Молочный коктейль": {"carbs_per_100ml": 10, "HE_per_100ml": 10 / 12},
    "Энергетик": {"carbs_per_100ml": 11, "HE_per_100ml": 11 / 12},
    "Тархун": {"carbs_per_100ml": 11.50, "HE_per_100ml": 11.50 / 12},
    "Сидр": {"carbs_per_100ml": 6, "HE_per_100ml": 6 / 12},
    "Швепс Тоник": {"carbs_per_100ml": 8.9, "HE_per_100ml": 8.9 / 12}
}

# Десерты
desserts_data = {
    "Тирамису": {"carbs_per_100g": 35},
    "Чизкейк": {"carbs_per_100g": 32},
    "Мороженое": {"carbs_per_100g": 24},
    "Шоколадный торт": {"carbs_per_100g": 45},
    "Пирожное картошка": {"carbs_per_100g": 50},
    "Наполеон": {"carbs_per_100g": 40, "HE_per_100g": 40 / 12},
    "Эклер": {"carbs_per_100g": 25, "HE_per_100g": 25 / 12},
    "Шоколадный торт": {"carbs_per_100g": 50, "HE_per_100g": 50 / 12},
    "Панкейки": {"carbs_per_100g": 36, "HE_per_100g": 36 / 12},
    "Мороженое": {"carbs_per_100g": 22, "HE_per_100g": 22 / 12},
    "Кекс": {"carbs_per_100g": 45, "HE_per_100g": 45 / 12}
    
}

# EfficientNet-B1
def load_model(weights_path, num_classes):
    model = efficientnet_b1(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Функция предсказания и расчёта
def predict_combined(image, food_grams, drink, drink_volume, sugar_spoons, include_dessert, dessert, dessert_grams):
    # Предсказание для еды
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    carbs_per_100g = carbohydrates_per_100g[predicted_class]
    total_carbohydrates_food = (food_grams / 100) * carbs_per_100g
    total_HE_food = total_carbohydrates_food / 12

    # Расчёты для напитка
    if drink in drinks_data:
        drink_data = drinks_data[drink]
        carbs_per_100ml = drink_data["carbs_per_100ml"]
        total_carbohydrates_drink = (drink_volume / 100) * carbs_per_100ml

        # с сахаром
        total_carbohydrates_sugar = sugar_spoons * 5  # 5 г углеводов на чайную ложку сахара
        total_carbohydrates_drink += total_carbohydrates_sugar
        total_HE_drink = total_carbohydrates_drink / 12
    else:
        carbs_per_100ml = 0
        total_carbohydrates_drink = 0
        total_HE_drink = 0

    # Расчёты для десертов (если включены)
    if include_dessert and dessert in desserts_data:
        dessert_data = desserts_data[dessert]
        carbs_per_100g_dessert = dessert_data["carbs_per_100g"]
        total_carbohydrates_dessert = (dessert_grams / 100) * carbs_per_100g_dessert
        total_HE_dessert = total_carbohydrates_dessert / 12
    else:
        carbs_per_100g_dessert = 0
        total_carbohydrates_dessert = 0
        total_HE_dessert = 0

    # Итоговые результаты
    total_carbohydrates = total_carbohydrates_food + total_carbohydrates_drink + total_carbohydrates_dessert
    total_HE = total_HE_food + total_HE_drink + total_HE_dessert

    return (
        f"Еда:\n"
        f"Предсказанный класс: {predicted_class}\n"
        f"Вес: {food_grams} г\n"
        f"Углеводы на 100 г: {carbs_per_100g:.2f} уг\n"
        f"Общие углеводы: {total_carbohydrates_food:.2f} уг\n"
        f"Общие ХЕ: {total_HE_food:.2f}\n\n"
        f"Напиток:\n"
        f"Выбранный напиток: {drink}\n"
        f"Углеводы на 100 мл: {carbs_per_100ml:.2f} уг\n"
        f"Объем: {drink_volume} мл\n"
        f"Количество сахара: {sugar_spoons} чайных ложек\n"
        f"Общие углеводы: {total_carbohydrates_drink:.2f} уг\n"
        f"Общие ХЕ: {total_HE_drink:.2f}\n\n"
        f"Десерт:\n"
        f"Выбранный десерт: {dessert if include_dessert else 'Без десерта'}\n"
        f"Вес десерта: {dessert_grams if include_dessert else 0} г\n"
        f"Углеводы на 100 г: {carbs_per_100g_dessert:.2f} уг\n"
        f"Общие углеводы: {total_carbohydrates_dessert:.2f} уг\n"
        f"Общие ХЕ: {total_HE_dessert:.2f}\n\n"
        f"Итог:\n"
        f"Суммарные углеводы: {total_carbohydrates:.2f} уг\n"
        f"Суммарные ХЕ: {total_HE:.2f}"
    )

# Загрузка весов модели
weights_path = "efficientnet_food_model.pth"
num_classes = len(class_names)
model = load_model(weights_path, num_classes)


# SARIMA
def process_glucose_data(file, xe=0):
    # Чтение файла
    data = pd.read_excel(file.name)

    # Переименовать колонки
    data.rename(columns={
        "Время": "Time",
        "Время создания/изменения": "Datetime",
        "Глюкоза сенсора (ммоль/л)": "Sensor Glucose"
    }, inplace=True)

    # время в datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # Убедится, что данные отсортированы
    data.sort_values('Datetime', inplace=True)

    # Отбор данных за последние 24 часа
    end_time = data['Datetime'].max()
    start_time = end_time - pd.Timedelta(hours=24)
    data_last_24h = data[(data['Datetime'] >= start_time) & (data['Datetime'] <= end_time)]

    # Интервал в 5 минут
    data_5min_24h = data_last_24h.set_index('Datetime').resample('5T').mean().reset_index()

    # Убирать пропуски
    data_5min_24h['Sensor Glucose'] = data_5min_24h['Sensor Glucose'].fillna(method='ffill')

    # SARIMA
    train = data_5min_24h[:-6]
    test = data_5min_24h[-6:]

    model = SARIMAX(train['Sensor Glucose'],
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 288),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    sarima_model = model.fit(disp=False)
    forecast = sarima_model.forecast(steps=6)

    # Влияние ХЕ (1 ХЕ ~ +2 ммоль/л)
    xe_effect = xe * 2
    final_forecast = forecast + xe_effect

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Sensor Glucose'], label="Тренировочные данные", color='blue')
    plt.plot(test.index, test['Sensor Glucose'], label="Истинные значения (30 минут)", color='green')
    plt.plot(test.index, forecast, label="Прогноз SARIMA (30 минут)", color='red', linestyle='--')
    plt.plot(test.index, final_forecast, label="Прогноз с учетом ХЕ (30 минут)", color='purple', linestyle='--')
    plt.axhline(y=4, color='orange', linestyle='--', label='Гипогликемия (<4 ммоль/л)')
    plt.axhline(y=10, color='orange', linestyle='--', label='Гипергликемия (>10 ммоль/л)')
    plt.fill_between(train.index, 4, 10, color='green', alpha=0.2, label='Норма (4-10 ммоль/л)')
    plt.xlabel("Время")
    plt.ylabel("Глюкоза сенсора (ммоль/л)")
    plt.title("Прогноз уровня глюкозы на следующие 30 минут")
    plt.legend()
    plt.grid()

    # Сохранить график
    plt.savefig('forecast.png')

    # Генерация рекомендаций
    last_forecast_value = final_forecast.iloc[-1]

    recommendations = generate_recommendations(last_forecast_value)

    return 'forecast.png', recommendations



def generate_recommendations(last_forecast_value):
    """
    Генерация рекомендаций на основе прогноза уровня сахара в крови.
    """

    recommendations = []

    # Целевой уровень глюкозы
    TARGET_GLUCOSE = 7  # ммоль/л

    # Коэффициент чувствительности к инсулину (ISF)
    ISF = 0.3  # ЕД инсулина на снижение 1 ммоль/л

    # Коэффициент углеводов (CARB_RATIO)
    CARB_RATIO = 10  # 10 г углеводов ≈ повышение глюкозы на 1 ммоль/л

    if last_forecast_value < 4:
        # Расчёт необходимого количества углеводов
        delta_glucose = TARGET_GLUCOSE - last_forecast_value
        carbs_needed = delta_glucose * CARB_RATIO

        recommendations.append(
            f"⚠️ Прогноз: {last_forecast_value:.2f} ммоль/л. Риск гипогликемии!\n"
            f"Рекомендуется принять примерно {carbs_needed:.2f} грамм углеводов, чтобы достичь уровня {TARGET_GLUCOSE} ммоль/л.\n"
            f"Проверьте рекомендации с врачом."
        )
    elif last_forecast_value > 10:
        # Расчёт необходимой дозы инсулина для гипергликемии
        delta_glucose = last_forecast_value - TARGET_GLUCOSE
        insulin_dose = delta_glucose * ISF

        if last_forecast_value <= 15:
            recommendations.append(
                f"⚠️ Прогноз: {last_forecast_value:.2f} ммоль/л. Риск гипергликемии!\n"
                f"Рекомендуется ввести {insulin_dose:.2f} ЕД инсулина для достижения уровня {TARGET_GLUCOSE} ммоль/л.\n"
                f"Проверьте дозу инсулина с врачом."
            )
        else:
            # Срочная рекомендация при опасной гипергликемии
            recommendations.append(
                f"⚠️ Прогноз: {last_forecast_value:.2f} ммоль/л.\n" 
                f"🆘 Опасная гипергликемия!\n"
                f"Необходимо срочно ввести {insulin_dose:.2f} ЕД инсулина для достижения уровня {TARGET_GLUCOSE} ммоль/л.\n"
                f"Проверьте дозу инсулина с врачом.\n"
                f"Обратитесь за медицинской помощью, если уровень сахара не снижается."
            )
    elif 4 <= last_forecast_value <= 10:
        # Уровень в норме
        recommendations.append(
            f"✅ Прогноз: {last_forecast_value:.2f} ммоль/л. Уровень сахара в норме."
        )

        # Добавление предупреждений для приближающихся значений
        if 9.5 < last_forecast_value <= 10:
            recommendations.append(
                f"ℹ️ Уровень сахара приближается к гипергликемии (близко к 10 ммоль/л).\n"
                f"Рекомендуется:\n"
                f"- Выпить стакан воды (250-500 мл) для выведения лишней глюкозы.\n"
                f"- Умеренная физическая активность (15-20 минут ходьбы), если нет противопоказаний."
            )
        elif 4 <= last_forecast_value < 4.5:
            delta_glucose = TARGET_GLUCOSE - last_forecast_value
            carbs_needed = delta_glucose * CARB_RATIO
            recommendations.append(
                f"ℹ️ Уровень сахара приближается к гипогликемии (близко к 4 ммоль/л).\n"
                f"Рекомендуется принять {carbs_needed:.2f} грамм углеводов, чтобы удержать уровень ближе к {TARGET_GLUCOSE} ммоль/л."
            )

    return recommendations




# ----- взаимодействия -----


def calculate_he(image, food_grams, drink, drink_volume, sugar_spoons, include_dessert, dessert, dessert_grams):
    global model  # Используем глобальную модель из первого кода
    # Здесь вызывается predict_combined из первого кода
    result = predict_combined(image, food_grams, drink, drink_volume, sugar_spoons, include_dessert, dessert, dessert_grams)
    # Полный отформатированный результат возвращается для отображения
    return result

# Функция для вызова второго кода с использованием значения ХЕ
def glucose_forecast(file, xe):
    # Вызываем process_glucose_data из второго кода
    forecast_image, recommendations = process_glucose_data(file, xe)
    return forecast_image, "\n".join(recommendations)

#----- Объединенный интерфейс -----
with gr.Blocks() as interface:
    gr.Markdown("## Калькулятор ХЕ: Расчет углеводов и хлебных единиц")
    gr.Markdown(
        "<b>ЗАМЕЧАНИЕ:</b> Система пока поддерживает распознавание только 9 классов еды: "
        "<br>[<b>Falafel</b>, <b>Hamburger</b>, <b>Hummus</b>, <b>Lasagna</b>, <b>Paella</b>, <b>Poutine</b>, <b>Ramen</b>, <b>Ravioli</b>, <b>Samosa</b>]"
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Ввод данных")
            image_input = gr.Image(label="Загрузите изображение блюда", type="numpy")
            food_grams = gr.Slider(1, 500, step=1, label="Вес еды (г)")
            drink = gr.Dropdown(choices=list(drinks_data.keys()), label="Выберите напиток")
            drink_volume = gr.Slider(50, 1000, step=50, label="Объем напитка (мл)")
            sugar_spoons = gr.Slider(0, 10, step=1, label="Количество сахара (чайных ложек)")
            include_dessert = gr.Checkbox(label="Включить десерт?")
            dessert = gr.Dropdown(choices=list(desserts_data.keys()), label="Выберите десерт")
            dessert_grams = gr.Slider(1, 500, step=1, label="Вес десерта (г)")
            calculate_button = gr.Button("Рассчитать ХЕ")
            # xe_output = gr.Number(label="Рассчитанное ХЕ")

        with gr.Column(scale=2):
            gr.Markdown("### Результат")
            he_result = gr.Textbox(label="Результат расчета", lines=30)

    calculate_button.click(
        calculate_he,
        inputs=[image_input, food_grams, drink, drink_volume, sugar_spoons, include_dessert, dessert, dessert_grams],
        outputs=he_result
    )


    with gr.Tab("Прогноз уровня глюкозы"):
        gr.HTML(
    """
    <p><b>ВНИМАНИЕ:</b> Для того чтобы обработать прогноз уровня глюкозы, нужно загрузить файл формата <b>.xlsx</b>, который должен быть в следующем составе:</p>
    <ul>
        <li><b>Время:</b> столбец с интервалами времени (например, 0:04:36)</li>
        <li><b>Время создания/изменения:</b> столбец с временной меткой (например, 1/11/2019 0:04)</li>
        <li><b>Глюкоза сенсора (ммоль/л):</b> столбец с уровнем глюкозы (например, 14.3)</li>
    </ul>
    <br>
    <p><b>Пример данных:</b></p>
    <table style="border: 1px solid black; border-collapse: collapse; width: 100%; text-align: center;">
        <thead style="background-color: #f2f2f2;">
            <tr>
                <th style="border: 1px solid black; padding: 8px;">Время</th>
                <th style="border: 1px solid black; padding: 8px;">Время создания/изменения</th>
                <th style="border: 1px solid black; padding: 8px;">Глюкоза сенсора (ммоль/л)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">0:04:36</td>
                <td style="border: 1px solid black; padding: 8px;">1/11/2019 0:04</td>
                <td style="border: 1px solid black; padding: 8px;">14.3</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">0:09:36</td>
                <td style="border: 1px solid black; padding: 8px;">1/11/2019 0:09</td>
                <td style="border: 1px solid black; padding: 8px;">14.4</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">0:14:36</td>
                <td style="border: 1px solid black; padding: 8px;">1/11/2019 0:14</td>
                <td style="border: 1px solid black; padding: 8px;">14.5</td>
            </tr>
            <tr>
                <td style="border: 1px solid black; padding: 8px;">0:19:36</td>
                <td style="border: 1px solid black; padding: 8px;">1/11/2019 0:19</td>
                <td style="border: 1px solid black; padding: 8px;">14.8</td>
            </tr>
        </tbody>
    </table>
    """
)



        file_input = gr.File(label="Загрузите файл .xlsx с данными уровня глюкозы")
        xe_input = gr.Number(label="Введите количество ХЕ (из расчета выше)")
        forecast_image = gr.Image(label="Прогноз уровня глюкозы")
        recommendations_output = gr.Textbox(label="Рекомендации", lines=5)

        forecast_button = gr.Button("Прогнозировать")
        forecast_button.click(
            glucose_forecast,
            inputs=[file_input, xe_input],
            outputs=[forecast_image, recommendations_output]
        )

interface.launch()

