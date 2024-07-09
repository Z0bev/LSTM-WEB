document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var instrument = document.getElementById('instrument').value;
    var timeframe = document.getElementById('timeframe').value;
    
    fetchPrediction(instrument, timeframe);
});

function fetchPrediction(instrument, timeframe) {
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            instrument: instrument,
            timeframe: timeframe
        })
    })
    .then(response => response.json())
    .then(data => {
        createTickerCard(instrument, data, timeframe);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function createTickerCard(instrument, data, timeframe) {
    const tickersContainer = document.getElementById('tickers-container');
    const tickerCard = document.createElement('div');
    tickerCard.className = 'ticker-card';
    tickerCard.innerHTML = `
        <div class="ticker-header">
            <h2>${instrument}</h2>
            <button class="close-button" onclick="closeTickerCard(this)">×</button>
        </div>
        <div class="price-info">
            <p>Current: $${parseFloat(data.current_price).toFixed(2)}</p>
            <p>Predicted (${timeframe}): $${parseFloat(data.prediction[data.prediction.length - 1]).toFixed(2)}</p>
            <p>Accuracy: ${data.accuracy}%</p>
        </div>
        <div id="plot-${instrument}" class="plot-container"></div>
    `;
    tickersContainer.prepend(tickerCard);

    createPlot(instrument, data.historical_prices, data.prediction, timeframe);
}

function closeTickerCard(button) {
    const card = button.closest('.ticker-card');
    card.remove();
}


function createPlot(instrument, historicalPrices, prediction, timeframe) {
    const historicalDates = historicalPrices.map(price => new Date(price.date));
    const lastHistoricalDate = historicalDates[historicalDates.length - 1];
    
    const predictionDates = [];
    for (let i = 0; i < prediction.length; i++) {
        const date = new Date(lastHistoricalDate);
        date.setDate(lastHistoricalDate.getDate() + i + 1);
        predictionDates.push(date);
    }

    const historicalTrace = {
        x: historicalDates,
        y: historicalPrices.map(price => price.close),
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Price',
        line: {color: '#1f77b4'}
    };

    const predictionTrace = {
        x: predictionDates,
        y: prediction,
        type: 'scatter',
        mode: 'lines',
        name: 'Predicted Price',
        line: {color: '#ff7f0e'} 
    };

    const layout = {
        title: `${instrument} (${timeframe})`,
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' },
        legend: {orientation: 'h', y: -0.2},
        height: 300,  // Reduce the height of the plot
        width: 400,   // Set a fixed width for the plot
        margin: { l: 50, r: 20, t: 40, b: 50 }  // Adjust margins
    };

    Plotly.newPlot(`plot-${instrument}`, [historicalTrace, predictionTrace], layout);
}