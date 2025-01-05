# masters_thesis


### Список использованных аниме для создания датасета (train)

- **Jujutsu Kaisen S2** - Серии: 1, 5, 8, 12, 16, 21
- **Oshi No Ko S2** - Серии: 1, 4, 8, 10
- **Dededede** - Серии: 4, 8, 11, 15
- **Dungeon Meshi** - Серии: 1, 3, 7, 11, 15, 19, 22, 24
- **Fruits Basket the Final** - Серии: 3, 8, 11, 13
- **Giji Harem** - Серии: 2, 5, 9, 11
- **Kaguya-sama wa Kokurasetai: Ultra Romantic** - Серии: 4, 6, 10, 13
- **Karasu wa Aruji o Erabanai**  - Серии: 5, 9, 12, 16
- **Kusuriya no Hitorigoto**  - Серии: 4, 7, 10, 14, 18, 21
- **Make Heroine ga Oosugiru!**   - Серии: 2, 5, 7, 10
- **Nige Jouzu no Wakagimi** - Серии: 1, 4, 6, 9
- **Re:Zero kara Hajimeru Isekai Seikatsu S3** - Серии: 1, 2, 3
- **Sousou no Frieren** - Серии: 4, 7, 10, 14, 18, 22, 25, 27
- **Tsue to Tsurugi no Wistoria** - Серии: 1, 4, 6, 9
- **Vinland Saga S2** - Серии: 3, 7, 12, 16, 19, 21


**Всего получилось 73 видео.**


#### Часть 1 (40 видео):
- 48,255 наиболее похожих кадров
- 4.7% удалено с помощью `image_filter_ResNet`
- 37.52% удалено с помощью `ImageDeduplicator`
- Время обработки: 1.5 часа

#### Часть 2 (33 видео):
- 46,283 наиболее похожих кадров
- 4.07% удалено с помощью `image_filter_ResNet`
- 44.37% удалено с помощью `ImageDeduplicator`
- Время обработки: 2 часа

**Итог:**  
59388 .png файлов, общий объём: 116 Гб

**val:**
- Dandadan - Серии: 3, 10
- Spy x Family - Серии: 2, 8
- Bocchi the Rock - Серии: 3, 8
- Tengoku Daimakyou - Серии: 1, 9
- Lycoris Recoil - Серии: 3, 12

Всего получилось 10 серий, итого удалено 447 (2.97%) + 4699 (36.31%) = 5146 из 12940 изображений.



**test:**
- One Piece Fan Letter - Серии: 1
- Dandadan - Серии: 6, 12
- Spy x Family - Серии: 5, 11
- Bocchi the Rock - Серии: 5, 11
- Tengoku Daimakyou - Серии: 5, 13
- Lycoris Recoil - Серии: 7

---


## Как подготовить данные

1. Перейдите в папку `dataset_preparation`.
2. Поместите видео в директорию `raw_video/`.
3. Запустите команду в консоли:
   
   ```bash
   python preprocess_data.py
5. Полученные изображения будут сохранены в папке `output_images/`.
6. Логи будут находиться в папке `logs/`.
