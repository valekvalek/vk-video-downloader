# VK Video Downloader

Установка:

    pip install -U yt-dlp
    # ffmpeg должен быть в PATH (Windows: winget install ffmpeg; macOS: brew install ffmpeg)

Запуск:

    python vk_downloader.py                       # окно (GUI)
    python vk_downloader.py https://vkvideo.ru/live-52344015_456244485 --list
    python vk_downloader.py https://vkvideo.ru/live-52344015_456244485 -q 720 -o ./videos

Ссылки вида `live-…`, `video-…`, `clip-…` приводятся к `https://vkvideo.ru/video-<owner>_<id>`.
Если видео закрытое — добавьте `--cookies-from-browser chrome`.
Если VK изменит API и загрузка сломается — `pip install -U yt-dlp`.

## GitHub Actions
Actions -> Download video -> Run workflow -> вставьте ссылку. Файл появится в Artifacts запуска.
