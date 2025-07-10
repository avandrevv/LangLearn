let isMenuOpen = false;

function toggleMenu() {
    const menu = document.getElementById('avatar-menu');
    isMenuOpen = !isMenuOpen;
    
    if (isMenuOpen) {
        menu.classList.add('show');
        menu.style.display = 'block';
        document.addEventListener('click', closeMenuOutside);
    } else {
        menu.classList.remove('show');
        setTimeout(() => {
            menu.style.display = 'none';
        }, 200);
        document.removeEventListener('click', closeMenuOutside);
    }
}

function closeMenuOutside(event) {
    const menu = document.getElementById('avatar-menu');
    const avatar = document.getElementById('user-avatar');
    
    if (!menu.contains(event.target) && event.target !== avatar) {
        menu.classList.remove('show');
        setTimeout(() => {
            menu.style.display = 'none';
        }, 200);
        isMenuOpen = false;
        document.removeEventListener('click', closeMenuOutside);
    }
}

// Показать индикатор до начала запроса
document.body.addEventListener('htmx:beforeRequest', () => {
    const indicator = document.querySelector('.htmx-indicator');
    if (indicator) {
      indicator.style.display = 'block';
    }
  });
  
  // Спрятать индикатор после завершения запроса
  document.body.addEventListener('htmx:afterRequest', () => {
    const indicator = document.querySelector('.htmx-indicator');
    if (indicator) {
      indicator.style.display = 'none';
    }
  });

// Добавляем обработчик событий после загрузки DOM
document.addEventListener('DOMContentLoaded', function() {
    const avatar = document.getElementById('user-avatar');
    if (avatar) {
        avatar.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleMenu();
        });
    }
});


// Автофокус на поле ввода
document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.querySelector('textarea');
    textarea.focus();
    
    // Сохраняем позицию скролла при обновлении ответа
    const responseContainer = document.getElementById('response-container');
    let scrollPosition = 0;
    
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.querySelector('form').requestSubmit();
        }
    });
});

// Очистка формы после успешной отправки и автофокус
document.body.addEventListener('htmx:afterRequest', (evt) => {
    if (evt.target.id === 'response-container' && evt.detail.successful) {
        const form = document.querySelector('form');
        form.reset();
        document.querySelector('textarea').focus();
        
        // Плавная прокрутка к новому ответу
        const responseContainer = document.getElementById('response-container');
        responseContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});

// Обработка ошибок
document.body.addEventListener('htmx:responseError', (evt) => {
    const responseContainer = document.getElementById('response-container');
    responseContainer.innerHTML = `
        <div style="color: red;">
            ⚠️ Ошибка при обработке запроса. Пожалуйста, попробуйте позже.
        </div>
    `;
});



// Индикатор ввода (опционально)
document.querySelector('textarea').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});