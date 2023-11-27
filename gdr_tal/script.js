document.addEventListener('DOMContentLoaded', function () {
    // Fetch JSON data
    fetch('gallery.json')
        .then(response => response.json())
        .then(data => {
            const gallery = document.getElementById('pdfGallery');
            data.forEach(item => {
                const img = document.createElement('img');
                img.src = item.img.replace("posters", "img").replace(".pdf", ".jpg");
                img.alt = item.title;
                img.title = item.title;
                img.addEventListener('click', () => openModal(item));
                gallery.appendChild(img);
            });
        });

    const modal = document.getElementById('modal');
    const modalPDF = document.getElementById('modalPDF');
    const miniatureImg = document.getElementById('miniatureImg');
    const caption = document.getElementById('caption');
    const closeBtn = document.getElementsByClassName('close')[0];

    function openModal(item) {
        modal.style.display = 'block';
        modalPDF.src = item.img;
        miniatureImg.src = item.img; // Set the miniature image source
        caption.innerHTML = `<h2>${item.title}</h2>`;
        if (item.authors) {
            caption.innerHTML += `<p><strong>Authors:</strong> ${item.authors}</p>`;
        }
        if (item.description) {
            caption.innerHTML += `<p>${item.description}</p>`;
        }
    }

    window.closeModal = function() {
        modal.style.display = 'none';
    };

    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });
});
