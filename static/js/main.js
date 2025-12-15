document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const calculateBtn = document.getElementById('calculateBtn');

    if (dropzone) {
        dropzone.addEventListener('click', () => {
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileName.innerText = e.target.files[0].name;
                fileName.classList.add("text-blue-600", "font-medium");

                selectedfile = e.target.files[0]; // selectedfile -> input file berupa jpg/png
            }
        });
    }


    if (calculateBtn) {
        calculateBtn.addEventListener('click', async () => {
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('resultState').classList.add('hidden');
            document.getElementById('loadingState').classList.remove('hidden');

            const fileInput = document.getElementById('fileInput');
            const size = document.getElementById('size').value;
            const lebarkain = document.getElementById('width').value;
            const qty = document.getElementById('qty').value;
            
            const formData = new FormData();
            if (fileInput.files[0]) {
                formData.append('image', fileInput.files[0]);
            }
            formData.append('size', size);
            formData.append('width', width);
            formData.append('qty', qty);

            try {
                // KIRIM DATA KE BACKEND
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    console.log(result.r)
                    document.getElementById('cuttinganKaos').innerText = result.prediction;
                    document.getElementById('resultTotal').innerText = result.total_meter + " M";
                    document.getElementById('resultPerPcs').innerText = result.per_pcs + " M";
                    
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('resultState').classList.remove('hidden');
                } else {
                    alert("Error: " + result.error);
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('emptyState').classList.remove('hidden');
                }

            } catch (error) {
                console.error('Error:', error);
                alert("Gagal terhubung ke AI.");
                document.getElementById('loadingState').classList.add('hidden');
            }
        });
    }
});