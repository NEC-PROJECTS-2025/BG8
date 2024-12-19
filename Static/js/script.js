
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-upload');
    const resultMessage = document.getElementById('result-message');

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent form refresh

        if (!fileInput.files.length) {
            resultMessage.textContent = 'Please select a file before submitting.';
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        resultMessage.textContent = 'Processing... Please wait.';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Prediction failed.');
            }

            const data = await response.json();
            resultMessage.textContent = `Prediction: ${data.prediction}`;
        } catch (error) {
            resultMessage.textContent = `Error: ${error.message}`;
        }
    });
});
