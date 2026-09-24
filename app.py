"""
WikiArt Multi-Task Streamlit Application (Русская локализация).
Совместное предсказание жанра/направления живописи и исторического века/эпохи
с визуализацией внимания нейросети (Grad-CAM).
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from data.transforms import get_val_transforms
from data.dataset import CENTURY_ORDER
from models.multi_head_art_net import MultiHeadArtNet
from models.gradcam import MultiTaskGradCAM, overlay_cam_on_image

# -----------------------------------------------------------------------------
# Конфигурация и названия классов по умолчанию
# -----------------------------------------------------------------------------
DEFAULT_GENRES = [
    "Импрессионизм", "Реализм", "Романтизм", "Экспрессионизм", "Постимпрессионизм",
    "Символизм", "Модерн (Ар-нуво)", "Барокко", "Абстрактный экспрессионизм", "Кубизм",
    "Примитивизм", "Живопись цветового поля", "Ренессанс (Возрождение)", "Минимализм", "Поп-арт",
    "Рококо", "Фовизм", "Пуантилизм", "Укиё-э"
]

DEFAULT_CENTURIES = [
    "До XV века", "XV век", "XVI век", "XVII век", "XVIII век", "XIX век", "XX век", "XXI век"
]

# Англо-русский маппинг для загруженных из модели классов
GENRE_RU_MAP = {
    "Impressionism": "Импрессионизм",
    "Realism": "Реализм",
    "Romanticism": "Романтизм",
    "Expressionism": "Экспрессионизм",
    "Post-Impressionism": "Постимпрессионизм",
    "Post Impressionism": "Постимпрессионизм",
    "Symbolism": "Символизм",
    "Art Nouveau": "Модерн (Ар-нуво)",
    "Baroque": "Барокко",
    "Abstract Expressionism": "Абстрактный экспрессионизм",
    "Cubism": "Кубизм",
    "Primitivism": "Примитивизм (Наив)",
    "Color Field": "Живопись цветового поля",
    "Renaissance": "Ренессанс (Возрождение)",
    "Minimalism": "Минимализм",
    "Pop Art": "Поп-арт",
    "Rococo": "Рококо",
    "Fauvism": "Фовизм",
    "Pointillism": "Пуантилизм",
    "Ukiyo-e": "Укиё-э",
    "Ukiyo e": "Укиё-э"
}

CENTURY_RU_MAP = {
    "Pre-XV": "До XV века",
    "XV": "XV век (Ранний Ренессанс)",
    "XVI": "XVI век (Высокое Возрождение / Маньеризм)",
    "XVII": "XVII век (Эпоха Барокко)",
    "XVIII": "XVIII век (Рококо и Классицизм)",
    "XIX": "XIX век (Романтизм, Реализм, Импрессионизм)",
    "XX": "XX век (Авангард и Модернизм)",
    "XXI": "XXI век (Современное искусство)"
}

CENTURY_DESCRIPTIONS = {
    "До XV века": "Средневековье и византийский период: готика, религиозная иконография, плоская перспектива.",
    "XV век": "15 век (Раннее Возрождение, Флоренция, гуманизм, открытие линейной перспективы).",
    "XVI век": "16 век (Высокое Возрождение и маньеризм: Леонардо да Винчи, Микеланджело, Рафаэль).",
    "XVII век": "17 век (Эпоха Барокко: драматическое светотеневое контрастирование кьяроскуро, Караваджо, Рембрандт).",
    "XVIII век": "18 век (Рококо и Просвещение: куртуазная элегантность, пастельная гамма, Фрагонар, Буше, Жак-Луи Давид).",
    "XIX век": "19 век (Революция в живописи: романтизм Делакруа, реализм Курбе, рождение пленэрного импрессионизма).",
    "XX век": "20 век (Авангардный взрыв: кубизм, экспрессионизм, абстракция, поп-арт, разрушение формы).",
    "XXI век": "21 век (Актуальное современное искусство, цифровые техники, концептуализм).",
}

GENRE_DESCRIPTIONS = {
    "Импрессионизм": "Характерен видимыми раздельной вибрацией мазков, открытой композицией, передачей естественного освещения и фиксацией мимолетного мгновения (Моне, Ренуар, Писсарро).",
    "Реализм": "Точное, правдивое и детализированное изображение действительности и повседневной жизни без идеализации или академической напыщенности (Курбе, Репин).",
    "Романтизм": "Приоритет эмоционального порыва, величие стихии, драматизм исторических коллизий и бунтарский дух (Делакруа, Фридрих, Гойя).",
    "Экспрессионизм": "Намеренная деформация пропорций и обостренная колористика для выражения тревоги, душевной боли и экзистенциального напряжения (Мунк, Кирхнер).",
    "Постимпрессионизм": "Отказ от сиюминутности ради монументального поиска глубинной сути: фактурный пастозный мазок, яркие локальные цвета и символизм (Ван Гог, Сезанн, Гоген).",
    "Барокко": "Высокая патетика, динамичные диагональные композиции, богатство палитры и глухой светотеневой контраст Tenebrism (Караваджо, Рубенс, Рембрандт).",
    "Кубизм": "Рассмотрение объектов с множества ракурсов одновременно, геометризация объемов, разложение формы на грани (Пикассо, Брак).",
    "Абстрактный экспрессионизм": "Спонтанный жест, действие на холсте (Action Painting), эмоциональное воздействие чистых цветовых полей без предметных образов (Поллок, Ротко).",
    "Ренессанс (Возрождение)": "Гармония золотого сечения, анатомическая точность, мягкая светотень sfumato и классический гуманизм (Да Винчи, Рафаэль).",
    "Модерн (Ар-нуво)": "Текучие волнообразные линии, органические растительные орнаменты, декоративная утонченность и витражная выразительность (Климт, Муха).",
    "Поп-арт": "Использование образов массовой культуры, комиксов, рекламных плакатов и тиражируемой шелкографии (Энди Уорхол, Лихтенштейн).",
    "Фовизм": "«Дикая» экспрессия чистых несмешанных цветов, отказ от светотени и линейной перспективы (Матисс, Дерен).",
    "Живопись цветового поля": "Масштабные плоскости однородного или пульсирующего цвета, вызывающие медитативное погружение (Марк Ротко).",
    "Рококо": "Изящество, пастельные тона, причудливые завитки рокайля, галантные сцены и придворная легкость (Буше, Ватто).",
    "Пуантилизм": "Оптическое смешение цветов: нанесение краски точечными мазками раздельного спектра (Сёра, Синьяк).",
}

# -----------------------------------------------------------------------------
# Настройка страницы Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WikiArt Multi-Head: Анализ картин и Grad-CAM",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #d946ef 0%, #8b5cf6 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Кэшированная загрузка модели
# -----------------------------------------------------------------------------
@st.cache_resource
def load_multi_task_model(checkpoint_path: str, backbone: str, device: str):
    device = torch.device(device)
    
    genres = DEFAULT_GENRES
    centuries = DEFAULT_CENTURIES
    
    meta_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "classes_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                raw_genres = meta.get("genres", genres)
                raw_centuries = meta.get("centuries", centuries)
                genres = [GENRE_RU_MAP.get(g, g) for g in raw_genres]
                centuries = [CENTURY_RU_MAP.get(c, c) for c in raw_centuries]
                backbone = meta.get("backbone", backbone)
        except Exception:
            pass

    model = MultiHeadArtNet(
        num_genres=len(genres),
        num_centuries=len(centuries),
        backbone_name=backbone,
        pretrained=True
    ).to(device)

    is_checkpoint_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            is_checkpoint_loaded = True
        except Exception as e:
            st.warning(f"Примечание: Чекпоинт не найден ({e}). Запущен демонстрационный режим на весах ConvNeXt.")

    model.eval()
    gradcam = MultiTaskGradCAM(model)
    return model, gradcam, genres, centuries, is_checkpoint_loaded


# -----------------------------------------------------------------------------
# Боковая панель параметров
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Параметры модели")

device_choice = st.sidebar.selectbox(
    "Вычислительное устройство",
    options=["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
    format_func=lambda x: "GPU (CUDA)" if x == "cuda" else "Процессор (CPU)",
    index=0
)

backbone_choice = st.sidebar.selectbox(
    "Архитектура бэкбона (Backbone)",
    options=["convnext_tiny", "convnext_small", "swin_t", "efficientnet_v2_s", "resnet50"],
    index=0
)

checkpoint_file = st.sidebar.text_input(
    "Путь к чекпоинту модели (.pt)",
    value="./checkpoints/best_multi_head_artnet.pt"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Объяснимый ИИ (Grad-CAM)")
cam_task = st.sidebar.radio("Что визуализировать:", ["Жанр / Направление", "Исторический век"])
cam_alpha = st.sidebar.slider("Прозрачность теплокарты", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Особенности архитектуры:**
- **Двуглавая сеть (Multi-Head)**: одновременный прогноз жанра и века.
- **ConvNeXt**: современные сверточные слои 7×7 и LayerNorm.
- **Focal Loss**: устранение дисбаланса редких стилей.
- **Grad-CAM**: подсветка мазков и композиционных зон.
""")

model, gradcam, genres, centuries, ckpt_loaded = load_multi_task_model(
    checkpoint_file, backbone_choice, device_choice
)

# -----------------------------------------------------------------------------
# Заголовок
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Классификатор живописи WikiArt Multi-Head</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Совместное распознавание художественного направления и исторической эпохи с визуализацией внимания нейросети (Grad-CAM)</div>',
    unsafe_allow_html=True
)

if not ckpt_loaded:
    st.info("💡 **Интерактивный демо-режим**: модель инициализирована с предобученным бэкбоном ConvNeXt. Вы можете загрузить картину или выбрать готовый шедевр для теста инференса и карты внимания Grad-CAM.")

# -----------------------------------------------------------------------------
# Загрузка или выбор пресета
# -----------------------------------------------------------------------------
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    uploaded_file = st.file_uploader(
        "Загрузите изображение картины (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "png", "jpeg", "webp"]
    )

with col_input2:
    preset_choice = st.selectbox(
        "Или выберите образец шедевра:",
        options=[
            "Не выбрано (Своя загрузка)",
            "Импрессионизм (Клод Моне — Кувшинки)",
            "Постимпрессионизм (Ван Гог — Звездная ночь)",
            "Ренессанс (Леонардо да Винчи — Сфумато)",
            "Кубизм (Пабло Пикассо — Синтетическая геометрия)",
            "Барокко (Караваджо — Кьяроскуро)",
            "Абстрактный экспрессионизм (Марк Ротко — Цветовое поле)"
        ]
    )

target_image: Optional[Image.Image] = None

def generate_preset_sample(name: str) -> Image.Image:
    w, h = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]

    if "Моне" in name:
        b = 140 + 60 * np.sin(x / 30.0) + 40 * np.cos(y / 25.0)
        g = 160 + 50 * np.cos(x / 40.0) + 30 * np.sin(y / 20.0)
        r = 60 + 40 * np.sin((x + y) / 40.0)
        img[:, :, 0] = np.clip(r, 0, 255)
        img[:, :, 1] = np.clip(g, 0, 255)
        img[:, :, 2] = np.clip(b, 0, 255)
    elif "Ван Гог" in name:
        dist = np.sqrt((x - 200)**2 + (y - 200)**2)
        angle = np.arctan2(y - 200, x - 200)
        swirl = np.sin(dist / 15.0 - angle * 4.0)
        b = 180 + 70 * swirl
        g = 80 + 40 * np.cos(angle * 6.0)
        r = 40 + 150 * (dist < 50) + 40 * np.sin(dist / 20.0)
        img[:, :, 0] = np.clip(r, 0, 255)
        img[:, :, 1] = np.clip(g, 0, 255)
        img[:, :, 2] = np.clip(b, 0, 255)
    elif "Пикассо" in name:
        pattern = ((x // 50) + (y // 50)) % 2
        diag = ((x + y) // 40) % 3
        img[:, :, 0] = np.clip(160 + pattern * 50 - diag * 30, 0, 255)
        img[:, :, 1] = np.clip(120 + pattern * 30 + diag * 20, 0, 255)
        img[:, :, 2] = np.clip(80 + diag * 50, 0, 255)
    elif "Караваджо" in name:
        spotlight = np.exp(-((x - 180)**2 + (y - 180)**2) / (2 * 90**2))
        img[:, :, 0] = np.clip(220 * spotlight + 15, 0, 255)
        img[:, :, 1] = np.clip(170 * spotlight + 10, 0, 255)
        img[:, :, 2] = np.clip(110 * spotlight + 5, 0, 255)
    elif "Ротко" in name:
        img[:160, :] = [210, 50, 40]
        img[160:200, :] = [120, 20, 30]
        img[200:, :] = [235, 120, 30]
    else:
        grad = (x + y) / (w + h)
        img[:, :, 0] = np.clip(180 - grad * 60, 0, 255)
        img[:, :, 1] = np.clip(140 - grad * 50, 0, 255)
        img[:, :, 2] = np.clip(100 - grad * 40, 0, 255)

    return Image.fromarray(img)


if uploaded_file is not None:
    try:
        target_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
elif preset_choice != "Не выбрано (Своя загрузка)":
    target_image = generate_preset_sample(preset_choice)


# -----------------------------------------------------------------------------
# Инференс и Grad-CAM
# -----------------------------------------------------------------------------
if target_image is not None:
    val_transform = get_val_transforms(img_size=224)
    input_tensor = val_transform(target_image).unsqueeze(0).to(device_choice)

    # 1. Прямой проход
    with torch.no_grad():
        genre_logits, century_logits = model(input_tensor)
        genre_probs = F.softmax(genre_logits, dim=-1)[0].cpu().numpy()
        century_probs = F.softmax(century_logits, dim=-1)[0].cpu().numpy()

    # Топ-1 предсказания
    top1_genre_idx = int(np.argmax(genre_probs))
    top1_genre_name = genres[top1_genre_idx]
    top1_genre_conf = float(genre_probs[top1_genre_idx])

    top1_century_idx = int(np.argmax(century_probs))
    top1_century_name = centuries[top1_century_idx]
    top1_century_conf = float(century_probs[top1_century_idx])

    # 2. Построение Grad-CAM
    task_name = "genre" if "Жанр" in cam_task else "century"
    target_idx = top1_genre_idx if task_name == "genre" else top1_century_idx

    try:
        cam_heatmap = gradcam.generate_cam(
            input_tensor=input_tensor,
            task=task_name,
            target_class_idx=target_idx
        )
        cam_blended = overlay_cam_on_image(
            original_img=target_image,
            cam=cam_heatmap,
            alpha=cam_alpha
        )
    except Exception as e:
        st.warning(f"Примечание Grad-CAM: {e}")
        cam_blended = target_image

    # Результаты
    st.markdown("---")
    st.subheader("🎯 Результаты распознавания")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Предсказанный жанр", top1_genre_name)
    with mcol2:
        st.metric("Уверенность жанра", f"{top1_genre_conf * 100:.1f}%")
    with mcol3:
        st.metric("Предсказанный век", top1_century_name)
    with mcol4:
        st.metric("Уверенность эпохи", f"{top1_century_conf * 100:.1f}%")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown("**Исходное полотно**")
        st.image(target_image, use_container_width=True)

    with vcol2:
        st.markdown(f"**Тепловая карта Grad-CAM ({cam_task})**")
        st.image(cam_blended, use_container_width=True)
        st.caption("Теплые оттенки (красный/желтый) показывают мазки, фактуру и контрасты, определившие выбор нейросети.")

    # Топ кандидаты
    st.markdown("---")
    st.subheader("📊 Распределение вероятностей по кандидатам")

    top_k = 4
    top_genre_indices = np.argsort(genre_probs)[::-1][:top_k]
    top_century_indices = np.argsort(century_probs)[::-1][:top_k]

    df_genre_top = pd.DataFrame({
        'Жанр': [genres[i] for i in top_genre_indices],
        'Вероятность': [float(genre_probs[i]) for i in top_genre_indices],
    })

    df_century_top = pd.DataFrame({
        'Век': [centuries[i] for i in top_century_indices],
        'Вероятность': [float(century_probs[i]) for i in top_century_indices],
    })

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("##### Топ вероятных направлений/жанров")
        if HAS_ALTAIR:
            chart_g = alt.Chart(df_genre_top).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                x=alt.X('Вероятность:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                y=alt.Y('Жанр:N', sort='-x'),
                color=alt.Color('Вероятность:Q', scale=alt.Scale(scheme='purples'), legend=None)
            ).properties(height=220)
            st.altair_chart(chart_g, use_container_width=True)
        else:
            st.bar_chart(df_genre_top.set_index('Жанр'))

    with bcol2:
        st.markdown("##### Топ вероятных исторических эпох")
        if HAS_ALTAIR:
            chart_c = alt.Chart(df_century_top).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                x=alt.X('Вероятность:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                y=alt.Y('Век:N', sort='-x'),
                color=alt.Color('Вероятность:Q', scale=alt.Scale(scheme='teals'), legend=None)
            ).properties(height=220)
            st.altair_chart(chart_c, use_container_width=True)
        else:
            st.bar_chart(df_century_top.set_index('Век'))

    # Искусствоведческая справка
    st.markdown("---")
    st.subheader("📚 Искусствоведческая справка")
    hcol1, hcol2 = st.columns(2)

    with hcol1:
        st.markdown(f"**О направлении ({top1_genre_name}):**")
        genre_info = GENRE_DESCRIPTIONS.get(top1_genre_name, "Художественное течение с характерной визуальной выразительностью, композиционными решениями и колоритом.")
        st.info(genre_info)

    with hcol2:
        st.markdown(f"**Об эпохе ({top1_century_name}):**")
        # Поиск по ключу или значению
        century_info = CENTURY_DESCRIPTIONS.get(top1_century_name, "Исторический период, ознаменовавший ключевые сдвиги в европейской и мировой художественной традиции.")
        st.info(century_info)

else:
    st.markdown("""
    ### 🖼️ Как начать работу:
    1. Перетащите изображение картины в поле загрузки слева.
    2. Либо выберите готовый образец шедевра в выпадающем списке.
    3. Исследуйте **тепловую карту Grad-CAM** в боковой панели, чтобы увидеть, какие мазки и участки полотна повлияли на решение нейросети!
    """)

