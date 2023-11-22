document.addEventListener('DOMContentLoaded', function () {
    // Fetch JSON data
    fetch('poster-db.json')
        .then(response => response.json())
        .then(data => {
            // Populate the gallery
            const gallery = document.querySelector('.gallery');
            data.forEach(item => {
                const img = document.createElement('img');
                img.src = item.img;
                img.alt = item.title;
                img.title = item.title;
                img.addEventListener('click', () => openModal(item));
                gallery.appendChild(img);
            });
        });

    // Get modal elements
    const modal = document.getElementById('modal');
    const modalImage = document.getElementById('modalImage');
    const caption = document.getElementById('caption');
    const closeBtn = document.getElementsByClassName('close')[0];

    // Open modal
    function openModal(item) {
        modal.style.display = 'block';
        modalImage.src = item.img;
        caption.innerHTML = `<h2>${item.title}</h2>`;
        if (item.authors) {
            caption.innerHTML += `<p><strong>Authors:</strong> ${item.authors}</p>`;
        }
        if (item.description) {
            caption.innerHTML += `<p>${item.description}</p>`;
        }
    }

    // Close modal
    closeBtn.addEventListener('click', () => modal.style.display = 'none');

    // Close modal on outside click
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});
