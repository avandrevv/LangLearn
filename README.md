# LangLearnAI — Платформа для изучения английского языка

LangLearnAI — это интерактивная обучающая платформа на **FastAPI**, которая сочетает в себе:  
- регистрацию и авторизацию пользователей,  
- админ‑панель для управления,  
- блог с публикациями,  
- чат‑ассистент на базе **Google Gemini**,  
- диагностические тесты для определения уровня,  
- генерацию персонализированных курсов и подтем,  
- итоговые тесты для закрепления знаний.  

---

## Возможности

- Регистрация и вход с JWT‑аутентификацией и сессиями  
- Профиль пользователя с курсами, тестами и настройками  
- Админ‑панель для управления пользователями  
- Блог: создание, редактирование и удаление статей  
- LangLearnAI‑чат: отвечает только на вопросы по английскому языку  
- Диагностический тест: определяет уровень (A2–C1) по теме  
- Курсы: автоматически формируются на основе ошибок и уровня  
- Подтемы: AI‑генерация словаря, грамматики, текста, заданий и мини‑теста  
- Финальный тест: закрепление знаний и завершение курса  

---

## Структура проекта

```bash
.
├── app.py                  # Основное приложение FastAPI
├── requirements.txt        # Зависимости
├── langlearn.db            # SQLite база данных
├── static/                 # CSS и JS
│   ├── css/main.css
│   └── js/app.js
├── templates/              # HTML-шаблоны (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── settings.html
│   ├── posts.html
│   ├── post_detail.html
│   ├── post_update.html
│   ├── create-article.html
│   ├── admin_users.html
│   ├── about.html
│   ├── langlearnAI.html
│   ├── my_courses.html
│   ├── course_overview.html
│   ├── subtopic_content.html
│   ├── topic_selection.html
│   ├── test_page.html
│   ├── test_results.html
│   ├── final_test.html
│   └── final_result.html
└── instance/langlearn.db   # Альтернативная БД
```

---

## ⚙️ Установка и запуск

```bash
# 1. Клонируем репозиторий
git clone https://github.com/username/langlearnai.git
cd langlearnai

# 2. Создаём виртуальное окружение и ставим зависимости
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# 3. Запускаем сервер
uvicorn app:app --reload

# 4. Открываем в браузере
http://127.0.0.1:8000
```

---

## Основные эндпоинты

| Маршрут                     | Метод | Описание |
|------------------------------|-------|----------|
| `/register`                  | GET/POST | Регистрация пользователя |
| `/login`                     | GET/POST | Авторизация |
| `/logout`                    | GET   | Выход |
| `/profile`                   | GET   | Профиль пользователя |
| `/settings`                  | GET/POST | Настройки |
| `/admin_users`               | GET   | Список пользователей (админ) |
| `/delete-user/{username}`    | DELETE | Удаление пользователя (админ) |
| `/posts`                     | GET   | Список статей |
| `/create-article`            | GET/POST | Создание статьи |
| `/posts/{id}`                | GET   | Просмотр статьи |
| `/posts/{id}/update`         | GET/POST | Редактирование статьи |
| `/posts/{id}/delete`         | GET   | Удаление статьи |
| `/langlearnai`               | GET   | Чат LangLearnAI |
| `/get_response`              | POST  | Ответ AI на вопрос |
| `/welcome`                   | GET   | Выбор темы для теста |
| `/start_test`                | POST  | Генерация теста |
| `/test_question/{id}/{num}`  | GET   | Вопрос теста |
| `/submit_answer/{id}/{num}`  | POST  | Ответ на вопрос |
| `/test_results/{id}`         | GET   | Результаты теста |
| `/create_course/{test_id}`   | POST  | Создание курса |
| `/course/{course_id}`        | GET   | Обзор курса |
| `/course/{id}/{subtopic}`    | GET   | Подтема курса |
| `/submit_subtopic_test/{id}` | POST  | Ответы на мини‑тест |
| `/final_test/{course_id}`    | GET   | Итоговый тест |
| `/submit_final_test/{id}`    | POST  | Завершение курса |

---

## Модели базы данных

- **User** — пользователи (логин, пароль, роль)  
- **Test** — диагностические тесты (вопросы, ответы, ошибки)  
- **Course** — курсы (тема, уровень, структура, прогресс)  
- **CourseSubtopic** — подтемы курса (контент, тесты, ошибки)  
- **Article** — статьи блога  

---

## Интеграция с AI

- Используется **Google Gemini** (`gemini-2.5-flash`)  
- Генерация:  
  - диагностических тестов,  
  - структуры курса,  
  - контента подтем,  
  - финального теста  
- Чат LangLearnAI отвечает **только на вопросы по английскому языку**  

---

## Безопасность

- JWT‑аутентификация  
- Сессии с `SessionMiddleware`  
- Хэширование паролей через `passlib[bcrypt]`  
- CSRF‑защита (`SameSite=lax`)  

---

## Планы по развитию

- Добавить API‑документацию (Swagger/OpenAPI)  
- Реализовать прогресс‑бар для курсов  
- Поддержка мультиязычности  
- Docker‑контейнеризация  
