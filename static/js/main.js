// JavaScript básico para la aplicación Flask

// Función que se ejecuta cuando la página está lista
document.addEventListener('DOMContentLoaded', function() {
    console.log('Aplicación ML Supervisado cargada correctamente');
    
    // Inicializar funciones
    initializeNavigation();
    initializeAnimations();
    initializeCopyButtons();
});

// Función para manejar la navegación móvil
function initializeNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
        
        // Cerrar menú al hacer click en un enlace
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
                hamburger.classList.remove('active');
            });
        });
    }
}

// Función para animaciones básicas al hacer scroll
function initializeAnimations() {
    // Crear observer para elementos que aparecen al hacer scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    
    // Observar elementos con clase 'animate-on-scroll'
    const animatedElements = document.querySelectorAll('.case-box, .number-box, .algorithm-card, .step');
    
    animatedElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
}

// Función para copiar texto al portapapeles
function initializeCopyButtons() {
    // Agregar botón de copiar a elementos de código si existen
    const codeBlocks = document.querySelectorAll('code, .keyword-tag');
    
    codeBlocks.forEach(block => {
        block.style.cursor = 'pointer';
        block.title = 'Click para copiar';
        
        block.addEventListener('click', function() {
            const text = this.textContent;
            copyToClipboard(text);
            showMessage('Copiado: ' + text);
        });
    });
}

// Función auxiliar para copiar texto
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback para navegadores más antiguos
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
    }
}

// Función para mostrar mensajes temporales
function showMessage(message) {
    // Crear elemento de mensaje
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 4px;
        z-index: 1000;
        font-size: 14px;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    document.body.appendChild(messageDiv);
    
    // Mostrar mensaje
    setTimeout(() => {
        messageDiv.style.opacity = '1';
    }, 100);
    
    // Ocultar y remover mensaje después de 3 segundos
    setTimeout(() => {
        messageDiv.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(messageDiv);
        }, 300);
    }, 3000);
}

// Función para suavizar el scroll a elementos
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth'
        });
    }
}

// Función para agregar efectos hover a las tarjetas
function addHoverEffects() {
    const cards = document.querySelectorAll('.case-box, .number-box, .algorithm-card, .tech-item');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

// Función para contar números de forma animada
function animateNumbers() {
    const numberElements = document.querySelectorAll('.number-box h3, .accuracy-big');
    
    numberElements.forEach(element => {
        const text = element.textContent;
        const number = parseFloat(text.replace(/[^0-9.]/g, ''));
        
        if (!isNaN(number) && number > 0) {
            let current = 0;
            const increment = number / 50;
            const timer = setInterval(() => {
                current += increment;
                if (current >= number) {
                    element.textContent = text; // Restaurar texto original
                    clearInterval(timer);
                } else {
                    // Mostrar número actual manteniendo el formato
                    if (text.includes('%')) {
                        element.textContent = Math.floor(current) + '%';
                    } else {
                        element.textContent = Math.floor(current);
                    }
                }
            }, 50);
        }
    });
}

// Función para lazy loading de imágenes (si las hay)
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// Función para manejar el cambio de tema (día/noche) - opcional
function initializeThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-theme');
            
            // Guardar preferencia
            const isDark = document.body.classList.contains('dark-theme');
            localStorage.setItem('darkTheme', isDark);
        });
        
        // Cargar preferencia guardada
        const savedTheme = localStorage.getItem('darkTheme');
        if (savedTheme === 'true') {
            document.body.classList.add('dark-theme');
        }
    }
}

// Función para validación simple de formularios (si los hay)
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    field.style.borderColor = '#dc3545';
                    isValid = false;
                } else {
                    field.style.borderColor = '#28a745';
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showMessage('Por favor completa todos los campos requeridos');
            }
        });
    });
}

// Función para mostrar/ocultar elementos con toggle
function initializeToggleElements() {
    const toggleButtons = document.querySelectorAll('[data-toggle]');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-toggle');
            const target = document.getElementById(targetId);
            
            if (target) {
                if (target.style.display === 'none' || !target.style.display) {
                    target.style.display = 'block';
                    this.textContent = this.textContent.replace('Mostrar', 'Ocultar');
                } else {
                    target.style.display = 'none';
                    this.textContent = this.textContent.replace('Ocultar', 'Mostrar');
                }
            }
        });
    });
}

// Función para manejar búsqueda en la página
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const searchableElements = document.querySelectorAll('.case-box, .algorithm-card');
            
            searchableElements.forEach(element => {
                const text = element.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    element.style.display = 'block';
                    element.style.opacity = '1';
                } else {
                    element.style.display = 'none';
                }
            });
            
            if (searchTerm === '') {
                searchableElements.forEach(element => {
                    element.style.display = 'block';
                    element.style.opacity = '1';
                });
            }
        });
    }
}

// Inicializar todas las funciones adicionales después de un breve delay
setTimeout(() => {
    addHoverEffects();
    animateNumbers();
    initializeLazyLoading();
    initializeThemeToggle();
    initializeFormValidation();
    initializeToggleElements();
    initializeSearch();
}, 1000);

// Función para debug - solo en desarrollo
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.log('Modo desarrollo activado');
    
    // Mostrar información útil para desarrollo
    window.debugInfo = function() {
        console.log('Páginas disponibles:');
        console.log('- Inicio: /');
        console.log('- Casos: /casos');
        console.log('- Detalle: /caso/1, /caso/2, /caso/3, /caso/4');
        console.log('- Metodología: /metodologia');
    };
}