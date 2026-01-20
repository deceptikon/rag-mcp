import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language


# === НАСТРОЙКИ ===

IGNORE_DIRS = {
    # Сборка и зависимости
    'node_modules', '.next', 'dist', 'build', '.vercel', 'node_modules',
    'venv', '.venv', '__pycache__', 'mediafiles', 'staticfiles', 'static',

    # Кэши инструментов
    '.ruff_cache', '.mypy_cache', '.pytest_cache', '.cursor', '.husky',
    '.git', '.github', '.vscode', '_TMP', '.brv', '.ci',

    # Тесты (на первом этапе лучше скрыть, чтобы не путать модель)
    'cypress', 'tests'
}

IGNORE_FILES = {
    # Секреты и конфиги окружения
    '.env', '.env.example', '.env.local', '.env.backup', '.env.production.local',
    'sample.env', 'db.sqlite3',

    # Лок-файлы (огромные и бесполезные для RAG)
    'yarn.lock', 'package-lock.json', 'uv.lock', 'tsconfig.strict.tsbuildinfo',

    # Конфиги инструментов (лучше убрать, чтобы не забивать контекст)
    '.eslintrc.json', '.prettierrc.json', '.eslintignore', '.prettierignore',
    '.cursorignore', '.gitignore', '.dockerignore', '.vercelignore',
    'docker-compose.yml', 'Dockerfile', '.flake8', '.ruffignore'
}
# Указываем, по каким заголовкам делить
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# Какие файлы мы хотим "читать"
ALLOWED_EXTENSIONS = {
    # Backend
    '.py': Language.PYTHON,
    # Frontend
    '.js': Language.JS,
    '.jsx': Language.JS,
    '.ts': Language.JS,
    '.tsx': Language.JS,
    # Docs & Configs
    '.md': Language.MARKDOWN,
    '.json': None, # Для конфигов, но аккуратно
    '.sql': None,
    '': None # Для файлов без расширения
}

def load_and_chunk_project(root_path):
    documents = []

    print(f"🚀 Начинаем сканирование: {os.path.abspath(root_path)}")

    for dirpath, dirnames, filenames in os.walk(root_path):
        # 1. Фильтрация папок (удаляем ненужные из обхода)
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            if filename in IGNORE_FILES:
                continue

            file_ext = os.path.splitext(filename)[1]
            if file_ext not in ALLOWED_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, root_path)

            # 2. Чтение файла
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Ошибка чтения {relative_path}: {e}")
                continue

            # Пропускаем пустые файлы
            if not content.strip():
                continue

            # 3. Выбор сплиттера в зависимости от языка
            language = ALLOWED_EXTENSIONS.get(file_ext)
            if file_ext == '.md':
                # ЭТАП 1: Режем по заголовкам
                md_header_chunks = md_header_splitter.split_text(content)

                # ЭТАП 2: Дорезаем слишком длинные секции (если внутри раздела 5000 символов)
                # и объединяем метаданные
                final_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = final_splitter.split_documents(md_header_chunks)

                # Добавляем свои метаданные (путь к файлу) к уже созданным заголовкам
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": relative_path,
                        "filename": filename,
                        "type": "documentation"
                    })

            if language:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=1000,
                    chunk_overlap=100
                )
            else:
                # Универсальный сплиттер для SQL, TXT и прочего
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

            # 4. Нарезка на чанки
            # Мы добавляем метаданные СРАЗУ, чтобы потом не потерять контекст
            chunks = splitter.create_documents(
                [content],
                metadatas=[{
                    "source": relative_path,
                    "filename": filename,
                    "extension": file_ext
                }]
            )

            documents.extend(chunks)
            print(f"✅ Обработан: {relative_path} -> {len(chunks)} чанков")

    return documents
