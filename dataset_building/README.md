# Master's Thesis

## Список использованных аниме для создания датасета (train)

- **Jujutsu Kaisen S2** - Серии: 1, 5, 8, 12, 16, 21
- **Oshi No Ko S2** - Серии: 1, 4, 8, 10
- **Dededede** - Серии: 4, 8, 11, 15
- **Dungeon Meshi** - Серии: 1, 3, 7, 11, 15, 19, 22, 24
- **Fruits Basket the Final** - Серии: 3, 8, 11, 13
- **Giji Harem** - Серии: 2, 5, 9, 11
- **Kaguya-sama wa Kokurasetai: Ultra Romantic** - Серии: 4, 6, 10, 13
- **Karasu wa Aruji o Erabanai** - Серии: 5, 9, 12, 16
- **Kusuriya no Hitorigoto** - Серии: 4, 7, 10, 14, 18, 21
- **Make Heroine ga Oosugiru!** - Серии: 2, 5, 7, 10
- **Nige Jouzu no Wakagimi** - Серии: 1, 4, 6, 9
- **Re:Zero kara Hajimeru Isekai Seikatsu S3** - Серии: 1, 2, 3
- **Sousou no Frieren** - Серии: 4, 7, 10, 14, 18, 22, 25, 27
- **Tsue to Tsurugi no Wistoria** - Серии: 1, 4, 6, 9
- **Vinland Saga S2** - Серии: 3, 7, 12, 16, 19, 21
- **Kimi ni Todoke S3** - Серии: 1, 2, 3, 4, 5
- **Pluto** - Серии: 1, 3, 6, 8
- **Karakai Jouzu no Takagi-san S3** - Серии: 1, 3, 4, 7, 10, 12
- **Kage no Jitsuryokusha ni Naritakute! S2** - Серии: 1, 3, 5, 8, 10, 12
- **Chi. Chikyuu no Undou ni Tsuite** - Серии: 1, 4, 6, 9, 12, 15
- **Kaijuu 8-gou** - Серии: 2, 6, 9, 12
- **Ore dake Level Up na Ken** - Серии: 1, 3, 5, 7, 9, 11
- **Kimetsu no Yaiba: Katanakaji no Sato-hen** - Серии: 1, 3, 5, 7, 9, 11
- **Ao Ashi** - Серии: 1, 5, 9, 12, 15, 18, 20, 22, 24
- **Dr. Stone: New World S3** - Серии: 1, 3, 5, 7, 9, 11
- **Horimiya: Piece** - Серии: 1, 3, 5, 7, 9, 11, 12, 13
- **Lycoris Recoil** - Серии: 1, 3, 5, 7, 9, 11, 12, 13

**Всего получилось 146 видео для train датасета.**

### Часть 1 (40 видео):
- 48,255 наиболее непохожих кадров
- 4.7% удалено с помощью `image_filter_ResNet`
- 37.52% удалено с помощью `ImageDeduplicator`

### Часть 2 (33 видео):
- 46,283 наиболее непохожих кадров
- 4.07% удалено с помощью `image_filter_ResNet`
- 44.37% удалено с помощью `ImageDeduplicator`

### Часть 3 (35 видео):
- 46,832 наиболее непохожих кадров
- 4.04% удалено с помощью `image_filter_ResNet`
- 41.56% удалено с помощью `ImageDeduplicator`

### Часть 4 (38 видео):
- 55,349 наиболее непохожих кадров
- 5.19% удалено с помощью `image_filter_ResNet`
- 36.82% удалено с помощью `ImageDeduplicator`

**Итог (train):**  
127.9 тысяч .png файлов, общим объёмом 351.79 Гб.


## Список использованных аниме для создания датасета (val)
- **Dandadan** - Серии: 3, 10
- **Spy x Family** - Серии: 2, 8
- **Bocchi the Rock** - Серии: 3, 8
- **Tengoku Daimakyou** - Серии: 1, 9
- **Lycoris Recoil** - Серии: 3, 12

**Результаты**
- 13,387 наиболее непохожих кадров
- 2.97% удалено с помощью `image_filter_ResNet`
- 36.28% удалено с помощью `ImageDeduplicator`

**Итог (val):**  
9.9 тысяч .png файлов, общим объёмом 37.9 Гб.


## Список использованных аниме для создания датасета (test)
- **One Piece Fan Letter** - Серии: 1
- **Dandadan** - Серии: 6, 12
- **Spy x Family** - Серии: 5, 11
- **Bocchi the Rock** - Серии: 5, 11
- **Tengoku Daimakyou** - Серии: 5, 13
- **Lycoris Recoil** - Серии: 7

**Результаты**
- 14,706 наиболее непохожих кадров
- 3.16% удалено с помощью `image_filter_ResNet`
- 33.70% удалено с помощью `ImageDeduplicator`

**Итог (val):**  
11.3 тысяч .png файлов, общим объёмом 37.42 Гб.


---

## Как подготовить данные

1. Перейдите в папку `dataset_preparation`.
2. Поместите видео в директорию `raw_video/`.
3. Запустите команду в консоли:
   
   ```bash
   python preprocess_data.py
5. Полученные изображения будут сохранены в папке `output_images/`.
6. Логи будут находиться в папке `logs/`.
