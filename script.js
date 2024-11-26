// Mengambil data cuaca dari API
async function getWeatherData() {
    const response = await fetch('https://api.open-meteo.com/v1/forecast?latitude=-8.7996&longitude=115.1711&current_weather=true');
    const data = await response.json();
    return data.current_weather;
}

// Membuat model TensorFlow
async function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [3] }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
    return model;
}

// Melatih model menggunakan data dari file JSON
async function trainModel(model) {
    // Memuat data dari file JSON
    const response = await fetch('trainingData.json'); // Ganti dengan path yang sesuai
    const trainingData = await response.json();

    const xs = tf.tensor2d(trainingData.map(d => [d.temperature, d.windspeed, d.weatherCode]));
    const ys = tf.tensor2d(trainingData.map(d => [d.label]));

    await model.fit(xs, ys, { epochs: 100 });
    console.log("Model trained!");
}

// Mendapatkan deskripsi cuaca berdasarkan kode cuaca
function getWeatherDescription(weatherCode) {
    // Clear
    if (weatherCode === 0 || weatherCode === 100) {
        return "Clear";
    }
    // Cloudy
    else if ((weatherCode >= 1 && weatherCode <= 3) || (weatherCode >= 101 && weatherCode <= 103)) {
        return "Cloudy";
    }
    // Fog/Haze/Smoke
    else if (weatherCode === 5 || weatherCode === 6 || (weatherCode >= 10 && weatherCode <= 12) || 
             (weatherCode >= 40 && weatherCode <= 50) || (weatherCode >= 130 && weatherCode <= 136)) {
        return "Fog/Haze/Smoke";
    }
    // Rain
    else if ((weatherCode >= 20 && weatherCode <= 29) || (weatherCode >= 50 && weatherCode <= 70) || 
             (weatherCode >= 120 && weatherCode <= 130) || (weatherCode >= 140 && weatherCode <= 150) || 
             (weatherCode >= 160 && weatherCode <= 170)) {
        return "Rain";
    }
    // Snow
    else if ((weatherCode >= 70 && weatherCode <= 80) || (weatherCode >= 170 && weatherCode <= 180)) {
        return "Snow";
    }
    // Thunderstorm
    else if (weatherCode === 17 || weatherCode === 29 || (weatherCode >= 95 && weatherCode < 100) || 
             (weatherCode >= 190 && weatherCode < 200)) {
        return "Thunderstorm";
    }
    // Dust/Sandstorm
    else if (weatherCode === 7 || weatherCode === 8 || weatherCode === 9 || (weatherCode >= 30 && weatherCode < 40)) {
        return "Dust/Sandstorm";
    }
    // Other
    else {
        return "Other";
    }
}

// Melakukan prediksi menggunakan model
async function makePrediction(model, weatherData) {
    const temperature = weatherData.temperature || 0; // Default ke 0 jika undefined
    const windspeed = weatherData.windspeed || 0;
    const weatherCode = weatherData.weathercode || 0;

    // Normalisasi data
    const inputData = tf.tensor2d([[temperature / 50, windspeed / 100, weatherCode / 100]], [1, 3]);
    
    const prediction = model.predict(inputData);
    const result = await prediction.data(); // Ambil nilai dari tensor

    if (isNaN(result[0])) {
        console.error("Prediction result is NaN. Check input data and model training.");
    }

    const weatherDescription = getWeatherDescription(weatherCode);

    document.getElementById('weatherInfo').innerText = 
        `Temperature: ${temperature}°C\nWindspeed: ${windspeed} km/h\nWeather Condition: ${weatherDescription}`;
    document.getElementById('result').innerText = 
        `Prediction: ${result[0] > 0.5 ? 'Safe to travel' : 'Not safe to travel'}\nPrediction Value: ${result[0]?.toFixed(4) || 'NaN'}`;
}

// Menambahkan event listener untuk tombol prediksi
document.getElementById('predictBtn').addEventListener('click', async () => {
    const weatherData = await getWeatherData(); // Mengambil data cuaca
    const model = await createModel();
    await trainModel(model); // Melatih model dengan data dari JSON
    await makePrediction(model, weatherData); // Melakukan prediksi dengan data cuaca
});
