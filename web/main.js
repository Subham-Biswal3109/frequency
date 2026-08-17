document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    const resultContainer = document.getElementById('result-container');
    const resultDisplay = document.getElementById('result-display');
    const errorAlert = document.getElementById('error-alert');

    // UI state: loading
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    resultContainer.classList.add('hidden');
    errorAlert.classList.add('hidden');

    // Gather input
    const payload = {
        start_frequency_mhz: parseFloat(document.getElementById('start_frequency_mhz').value),
        end_frequency_mhz: parseFloat(document.getElementById('end_frequency_mhz').value),
        bandwidth_mhz: parseFloat(document.getElementById('bandwidth_mhz').value),
        service_type: document.getElementById('service_type').value,
        state: document.getElementById('state').value,
        city: document.getElementById('city').value,
        hour_of_day: parseInt(document.getElementById('hour_of_day').value),
        day_of_week: parseInt(document.getElementById('day_of_week').value),
        signal_power_dbm: parseFloat(document.getElementById('signal_power_dbm').value),
        noise_floor_dbm: parseFloat(document.getElementById('noise_floor_dbm').value),
        snr_db: parseFloat(document.getElementById('snr_db').value)
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Render success
        const isAvailable = data.available;
        const probability = data.probability !== null ? (data.probability * 100).toFixed(1) + '%' : 'N/A';
        
        const badgeClass = isAvailable ? 'status-available' : 'status-unavailable';
        const badgeText = isAvailable ? 'Available' : 'Unavailable';
        
        resultDisplay.innerHTML = `
            <div class="status-badge ${badgeClass}">${badgeText}</div>
            <div style="color: var(--text-muted); margin-top: 1rem;">Confidence Score</div>
            <div class="prob-circle" style="border-color: ${isAvailable ? 'var(--success)' : 'var(--danger)'}">
                ${probability}
            </div>
            <p style="margin-top: 1rem; color: var(--text-muted); font-size: 0.9rem;">Prediction has been logged to the database.</p>
        `;
        resultContainer.classList.remove('hidden');
        
    } catch (err) {
        // Render error
        errorAlert.textContent = err.message;
        errorAlert.classList.remove('hidden');
    } finally {
        // Restore UI
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
