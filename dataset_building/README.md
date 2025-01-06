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
- Kimi ni Todoke S3 - Серии: 1,2,3,4,5
- Pluto - Серии: 1,3,6,8
- Karakai Jouzu no Takagi-san S3 - Серии: 1,3,4,7,10,12
- Kage no Jitsuryokusha ni Naritakute! S2 - Серии: 1,3,5,8,10,12
- Chi. Chikyuu no Undou ni Tsuite - Серии: 1,4,6,9,12,15
- Kaijuu 8-gou - Серии: 2,6,9,12
- Ore dake Level Up na Ken - Серии: 1,3,5,7,9,11
- Kimetsu no Yaiba: Katanakaji no Sato-hen - Серии: 1,3,5,7,9,11
- Ao Ashi - Серии: 1,5,9,12,15,18,20,22,24
- Dr. Stone: New World S3 - Серии: 1,3,5,7,9,11
- Horimiya: Piece - Серии: 1,3,5,7,9,11,12,13
- Lycoris Recoil - Серии: 1,3,5,7,9,11,12,13

**Всего получилось 146 видео.**

#### Часть 1 (40 видео):
- 48,255 наиболее непохожих кадров
- 4.7% удалено с помощью `image_filter_ResNet`
- 37.52% удалено с помощью `ImageDeduplicator`
- Время обработки: 1.5 часа

#### Часть 2 (33 видео):
- 46,283 наиболее непохожих кадров
- 4.07% удалено с помощью `image_filter_ResNet`
- 44.37% удалено с помощью `ImageDeduplicator`
- Время обработки: 2 часа

Часть 3 (35 видео):
- 32 697 наиболее непохожих кадров
- 2025-01-06 19:14:51,999 - ImageDeduplicator - INFO - Удалено 18569 дубликатов из 44675 изображений. Доля удаленных: 41.56%
- 2025-01-06 18:14:38,168 - tools.image_filter_ResNet - INFO - Selected 2157 images for removal based on evaluation scores.
- 2025-01-06 18:14:38,169 - tools.image_filter_ResNet - INFO - Fraction of images removed: 4.04%
Время обработки: 2 часа 45 минут


Часть 4 (38 видео):


**Итог:**  
59388 .png файлов, общий объём: 116 Гб

**val:**
- Dandadan - Серии: 3, 10
- Spy x Family - Серии: 2, 8
- Bocchi the Rock - Серии: 3, 8
- Tengoku Daimakyou - Серии: 1, 9
- Lycoris Recoil - Серии: 3, 12

2025-01-05 21:58:40,125 - tools.image_filter_ResNet - INFO - Selected 447 images for removal based on evaluation scores.
2025-01-05 21:58:40,125 - tools.image_filter_ResNet - INFO - Fraction of images removed: 2.97%
 ImageDeduplicator - INFO - Удалено 4694 дубликатов из 12940 изображений. Доля удаленных: 36.28%



**test:**
- One Piece Fan Letter - Серии: 1
- Dandadan - Серии: 6, 12
- Spy x Family - Серии: 5, 11
- Bocchi the Rock - Серии: 5, 11
- Tengoku Daimakyou - Серии: 5, 13
- Lycoris Recoil - Серии: 7

2025-01-05 23:04:18,882 - tools.image_filter_ResNet - INFO - Selected 524 images for removal based on evaluation scores.
2025-01-05 23:04:18,883 - tools.image_filter_ResNet - INFO - Fraction of images removed: 3.16%
2025-01-05 23:13:51,474 - ImageDeduplicator - INFO - Удалено 4779 дубликатов из 14182 изображений. Доля удаленных: 33.70%

---


## Как подготовить данные

1. Перейдите в папку `dataset_preparation`.
2. Поместите видео в директорию `raw_video/`.
3. Запустите команду в консоли:
   
   ```bash
   python preprocess_data.py
5. Полученные изображения будут сохранены в папке `output_images/`.
6. Логи будут находиться в папке `logs/`.
