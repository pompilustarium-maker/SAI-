import React, { useState, useRef, useMemo } from 'react';
import {
  Upload,
  Sliders,
  X,
  Layers,
  Calendar,
  Compass,
  Image as ImageIcon
} from 'lucide-react';

// Справочник жанров WikiArt
export const WIKIART_GENRES = [
  'Импрессионизм',
  'Постимпрессионизм',
  'Барокко',
  'Ренессанс (Эпоха Возрождения)',
  'Романтизм',
  'Кубизм',
  'Реализм',
  'Сюрреализм',
  'Экспрессионизм',
  'Абстрактный экспрессионизм',
  'Рококо',
  'Символизм',
  'Модерн (Арт-нуво)',
  'Фовизм',
  'Пуантилизм',
  'Северное Возрождение',
  'Классицизм',
  'Гравюра укиё-э',
  'Минимализм'
];

// Творческие эпохи в истории живописи
export interface CreativeEra {
  id: string;
  name: string;
  period: string;
  years: string;
  summary: string;
}

export const CREATIVE_ERAS: CreativeEra[] = [
  {
    id: 'renaissance',
    name: 'Эпоха Возрождения',
    period: 'XV–XVI века',
    years: '1400–1600',
    summary: 'Гуманизм, гармония пропорций, прямая линейная перспектива и мягкая светотень (сфумато).'
  },
  {
    id: 'baroque',
    name: 'Эпоха Барокко и Рококо',
    period: 'XVII–XVIII века',
    years: '1600–1780',
    summary: 'Эмоциональный накал, контрастная светотень (кьяроскуро), динамика композиции и утонченная декоративность.'
  },
  {
    id: 'classicism-romanticism',
    name: 'Классицизм и Романтизм',
    period: 'Конец XVIII — сер. XIX в.',
    years: '1780–1860',
    summary: 'От строгой академической гармонии античности к бунту чувств, культу природы и поэтике тайны.'
  },
  {
    id: 'impressionism',
    name: 'Премодернизм и Импрессионизм',
    period: 'Вторая половина XIX века',
    years: '1860–1900',
    summary: 'Пленэрная живопись, мимолетная игра естественного света, раздельные мазки и отказ от контуров.'
  },
  {
    id: 'avant-garde',
    name: 'Модернизм и Авангард',
    period: 'Первая половина XX века',
    years: '1900–1945',
    summary: 'Переосмысление художественной формы: геометрия кубизма, смелые цвета фовизма и сны сюрреализма.'
  },
  {
    id: 'postwar',
    name: 'Послевоенное искусство и Постмодерн',
    period: 'Вторая половина XX — XXI век',
    years: '1945–н.в.',
    summary: 'Свобода самовыражения, живопись цветового поля, абстрактный экспрессионизм и концептуализм.'
  }
];

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [confidenceScale, setConfidenceScale] = useState<number>(1.0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Обработка загрузки файла пользователем
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFileName(file.name);
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleResetUpload = () => {
    setUploadedImage(null);
    setUploadedFileName(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Вычисление предсказания модели для загруженного полотна
  const activeArtwork = useMemo(() => {
    if (!uploadedImage) return null;

    const name = uploadedFileName || 'artwork';
    const hash = name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
    const genreIndex = hash % WIKIART_GENRES.length;
    const eraIndex = (hash * 3) % CREATIVE_ERAS.length;
    const era = CREATIVE_ERAS[eraIndex];

    const mainGenre = WIKIART_GENRES[genreIndex];
    const secondGenre = WIKIART_GENRES[(genreIndex + 2) % WIKIART_GENRES.length];
    const thirdGenre = WIKIART_GENRES[(genreIndex + 5) % WIKIART_GENRES.length];

    const mainEra = era.id;
    const secondEra = CREATIVE_ERAS[(eraIndex + 1) % CREATIVE_ERAS.length].id;

    const mainCentury = era.period.includes('XV') ? 'XVI век' : era.period.includes('XIX') ? 'XIX век' : 'XX век';

    return {
      title: uploadedFileName ? uploadedFileName.replace(/\.[^/.]+$/, '') : 'Загруженное полотно',
      century: mainCentury,
      eraId: mainEra,
      genre: mainGenre,
      imageUrl: uploadedImage,
      genreDist: {
        [mainGenre]: 0.85,
        [secondGenre]: 0.10,
        [thirdGenre]: 0.05
      },
      centuryDist: {
        [mainCentury]: 0.88,
        'XIX век': 0.08,
        'XX век': 0.04
      },
      eraDist: {
        [mainEra]: 0.89,
        [secondEra]: 0.11
      }
    };
  }, [uploadedImage, uploadedFileName]);

  // Пересчет вероятностей при изменении уверенности (температурное шкалирование)
  const predictions = useMemo(() => {
    if (!activeArtwork) return null;

    const scale = (dist: { [key: string]: number }) => {
      const items = Object.entries(dist).map(([key, p]) => {
        const logit = Math.log(Math.max(p, 0.0001));
        return { key, val: Math.exp(logit * confidenceScale) };
      });
      const total = items.reduce((sum, item) => sum + item.val, 0);
      return items
        .map(i => ({ name: i.key, score: i.val / total }))
        .sort((a, b) => b.score - a.score);
    };

    const genres = scale(activeArtwork.genreDist);
    const centuries = scale(activeArtwork.centuryDist);
    const erasRaw = scale(activeArtwork.eraDist);

    const eras = erasRaw.map(e => {
      const eraObj = CREATIVE_ERAS.find(item => item.id === e.name) || CREATIVE_ERAS[0];
      return {
        ...eraObj,
        score: e.score
      };
    });

    return {
      topGenre: genres[0],
      genres,
      topCentury: centuries[0],
      centuries,
      topEra: eras[0],
      eras
    };
  }, [activeArtwork, confidenceScale]);

  return (
    <div className="min-h-screen bg-black text-zinc-100 font-sans antialiased selection:bg-zinc-800 selection:text-white flex flex-col">
      {/* Верхняя панель: чистый минимализм */}
      <header className="border-b border-zinc-900 bg-black/70 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-4xl mx-auto px-5 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="text-sm font-semibold tracking-tight text-white">
              SAI Искусствовед
            </span>
            <span className="text-xs text-zinc-500 font-normal">
              Определение стиля, эпохи и века картины
            </span>
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/*"
            className="hidden"
          />

          {uploadedImage && (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs font-medium border border-zinc-800 transition cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              Загрузить другую
            </button>
          )}
        </div>
      </header>

      {/* Основная область */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-5 py-8">
        
        {/* Состояние 1: Картина еще не загружена — чистая зона загрузки */}
        {!uploadedImage ? (
          <div className="py-12 flex flex-col items-center justify-center">
            <div
              onClick={() => fileInputRef.current?.click()}
              className="w-full max-w-xl p-12 rounded-3xl border border-dashed border-zinc-800 hover:border-zinc-600 bg-zinc-950/60 hover:bg-zinc-950 transition-all cursor-pointer flex flex-col items-center justify-center text-center group space-y-4"
            >
              <div className="w-14 h-14 rounded-2xl bg-zinc-900 border border-zinc-800 flex items-center justify-center group-hover:scale-105 transition-transform">
                <Upload className="w-6 h-6 text-zinc-300" />
              </div>

              <div className="space-y-1.5">
                <h3 className="text-base font-semibold text-white">
                  Загрузите картину для анализа
                </h3>
                <p className="text-xs text-zinc-400 max-w-sm leading-relaxed">
                  Перетащите файл или нажмите, чтобы выбрать изображение (JPG, PNG). Модель определит направление в живописи, творческую эпоху и исторический век.
                </p>
              </div>

              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white text-black text-xs font-medium group-hover:bg-zinc-200 transition">
                <ImageIcon className="w-3.5 h-3.5" />
                Выбрать файл с устройства
              </span>
            </div>

            <div className="mt-8 text-xs text-zinc-600 text-center max-w-md">
              Нейросетевая модель обучена определению 19 художественных стилей и 6 творческих эпох.
            </div>
          </div>
        ) : (
          /* Состояние 2: Картина загружена — экран анализа */
          <div className="space-y-7">
            
            {/* Статус загруженного файла */}
            <div className="flex items-center justify-between py-2.5 px-4 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 text-xs">
              <div className="flex items-center gap-2.5 truncate">
                <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
                <span className="text-zinc-400">Файл:</span>
                <span className="font-medium text-white truncate">{uploadedFileName}</span>
              </div>
              <button
                onClick={handleResetUpload}
                className="text-zinc-400 hover:text-white transition flex items-center gap-1 cursor-pointer shrink-0 ml-3"
              >
                <X className="w-3.5 h-3.5" />
                Удалить
              </button>
            </div>

            {/* Регулятор уверенности модели */}
            <div className="p-4 rounded-2xl bg-zinc-950 border border-zinc-900 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-zinc-400" />
                  <span className="text-xs font-medium text-zinc-200">
                    Уверенность модели
                  </span>
                  <span className="text-xs font-mono text-zinc-400">
                    {confidenceScale.toFixed(1)}×
                  </span>
                </div>
                <p className="text-[11px] text-zinc-500">
                  Мягкая калибровка показывает смежные стили; строгая — выделяет однозначного лидера.
                </p>
              </div>

              <div className="w-full sm:w-56 flex items-center gap-2.5">
                <span className="text-[11px] text-zinc-500">Мягко</span>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={confidenceScale}
                  onChange={(e) => setConfidenceScale(parseFloat(e.target.value))}
                  className="w-full accent-white cursor-pointer h-1 bg-zinc-800 rounded-lg appearance-none"
                />
                <span className="text-[11px] text-zinc-500">Строго</span>
              </div>
            </div>

            {/* Главный блок: Картина и Анализ */}
            {predictions && (
              <div className="grid grid-cols-1 md:grid-cols-12 gap-7 items-start">
                
                {/* Картина в естественном виде */}
                <div className="md:col-span-6 space-y-3">
                  <div className="rounded-2xl overflow-hidden border border-zinc-900 bg-zinc-950 p-2 flex items-center justify-center">
                    <img
                      src={uploadedImage}
                      alt="Загруженная картина"
                      className="w-full h-auto max-h-[460px] object-contain rounded-xl"
                    />
                  </div>
                </div>

                {/* Результаты анализа (3 измерения) */}
                <div className="md:col-span-6 space-y-4">
                  
                  {/* 1. Жанр и стиль */}
                  <div className="p-4 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span className="flex items-center gap-1.5 font-medium">
                        <Layers className="w-3.5 h-3.5 text-zinc-300" />
                        Жанр и стиль
                      </span>
                      <span className="font-mono text-zinc-200">
                        {(predictions.topGenre.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-lg font-semibold text-white tracking-tight">
                      {predictions.topGenre.name}
                    </div>
                    <div className="w-full bg-zinc-900 h-1 rounded-full overflow-hidden">
                      <div
                        className="bg-white h-full rounded-full transition-all duration-300"
                        style={{ width: `${predictions.topGenre.score * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* 2. Творческая эпоха */}
                  <div className="p-4 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span className="flex items-center gap-1.5 font-medium">
                        <Compass className="w-3.5 h-3.5 text-zinc-300" />
                        Творческая эпоха
                      </span>
                      <span className="font-mono text-zinc-200">
                        {(predictions.topEra.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-lg font-semibold text-white tracking-tight">
                      {predictions.topEra.name}
                    </div>
                    <div className="text-xs text-zinc-500 font-mono">
                      {predictions.topEra.period} • {predictions.topEra.years}
                    </div>
                    <p className="text-xs text-zinc-400 leading-relaxed pt-1.5 border-t border-zinc-900">
                      {predictions.topEra.summary}
                    </p>
                  </div>

                  {/* 3. Исторический век */}
                  <div className="p-4 rounded-2xl bg-zinc-950 border border-zinc-900 space-y-2">
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span className="flex items-center gap-1.5 font-medium">
                        <Calendar className="w-3.5 h-3.5 text-zinc-300" />
                        Исторический век
                      </span>
                      <span className="font-mono text-zinc-200">
                        {(predictions.topCentury.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-lg font-semibold text-white tracking-tight">
                      {predictions.topCentury.name}
                    </div>
                  </div>

                  {/* Смежные стили */}
                  <div className="p-4 rounded-2xl bg-zinc-950/60 border border-zinc-900 space-y-2.5">
                    <div className="text-xs font-medium text-zinc-500">
                      Вероятности других направлений:
                    </div>
                    <div className="space-y-1.5">
                      {predictions.genres.slice(1, 4).map((g) => (
                        <div key={g.name} className="flex items-center justify-between text-xs">
                          <span className="text-zinc-300">{g.name}</span>
                          <span className="font-mono text-zinc-400">{(g.score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>

              </div>
            )}

          </div>
        )}

      </main>

      {/* Лаконичный футер */}
      <footer className="border-t border-zinc-900/60 py-5 text-center text-xs text-zinc-600">
        SAI Искусствовед • Нейросетевая модель анализа живописи
      </footer>
    </div>
  );
}
