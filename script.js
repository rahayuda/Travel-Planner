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
    switch (weatherCode) {
        case 0: return "Clear sky";
        case 1: return "Mainly clear";
        case 2: return "Partly cloudy"; // Menambahkan deskripsi cuaca
        case 3: return "Overcast";
        case 45: return "Fog";
        case 51: return "Light rain";
        case 61: return "Moderate rain";
        case 71: return "Heavy rain";
        case 80: return "Snow showers";
        case 95: return "Thunderstorms";
        default: return "Unknown weather condition";
    }
}

// Melakukan prediksi menggunakan model
async function makePrediction(model, weatherData) {
    const temperature = weatherData.temperature; // Suhu
    const windspeed = weatherData.windspeed; // Kecepatan angin
    const weatherCode = weatherData.weathercode; // Kode cuaca

    const inputData = tf.tensor2d([[temperature, windspeed, weatherCode]], [1, 3]);
    const prediction = model.predict(inputData);
    const result = await prediction.data();

    // Mendapatkan deskripsi cuaca berdasarkan kode cuaca
    const weatherDescription = getWeatherDescription(weatherCode);

    // Menampilkan informasi cuaca
    document.getElementById('weatherInfo').innerText = `Temperature: ${temperature}°C\nWindspeed: ${windspeed} km/h\nWeather Condition: ${weatherDescription}`;

    // Menampilkan hasil prediksi
    document.getElementById('result').innerText = `Prediction: ${result[0] > 0.4 ? 'Safe to travel' : 'Not safe to travel'}`;
}

// Menambahkan event listener untuk tombol prediksi
document.getElementById('predictBtn').addEventListener('click', async () => {
    const weatherData = await getWeatherData(); // Mengambil data cuaca
    const model = await createModel();
    await trainModel(model); // Melatih model dengan data dari JSON
    await makePrediction(model, weatherData); // Melakukan prediksi dengan data cuaca
});
