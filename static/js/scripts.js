window.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('floatingImages');
    const imageNames = ['emotion_0012.jpg', 'emotion_0031.jpg', 'emotion_0090.jpg', 'emotion_0157.jpg', 'emotion_0268.jpg'];
    for (let i = 0; i < 50; i++) {  // Creating multiple images for a denser effect
        const img = document.createElement('img');
        img.src = `/static/images/${imageNames[i % imageNames.length]}`;
        img.className = 'floating-img';
        img.style.top = `${Math.random() * 100}%`;
        img.style.left = `${Math.random() * 100}%`;
        container.appendChild(img);
    }
});
