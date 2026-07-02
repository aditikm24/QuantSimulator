document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('simulation-form');
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    
    // UI Panels
    const optionsResults = document.getElementById('options-results');
    const portfolioResults = document.getElementById('portfolio-results');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI State: Loading
        btnText.style.display = 'none';
        loader.style.display = 'block';
        optionsResults.style.display = 'none';
        portfolioResults.style.display = 'none';
        document.getElementById('simulate-btn').disabled = true;

        const ticker = document.getElementById('ticker').value;
        const historicalDays = parseInt(document.getElementById('historical-days').value);
        const simulationDays = parseInt(document.getElementById('simulation-days').value);
        const model = document.getElementById('model-type').value;
        const strikePriceInput = document.getElementById('strike-price').value;
        const strikePrice = strikePriceInput ? parseFloat(strikePriceInput) : null;

        const requestData = {
            ticker,
            historical_days: historicalDays,
            simulation_days: simulationDays,
            model,
            num_simulations: 100, // Reduced from 1000 for browser memory safety, enough for VaR demo
            strike_price: strikePrice
        };

        try {
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errData = await response.json();
                const errMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
                throw new Error(errMsg || 'Simulation failed');
            }

            const data = await response.json();
            
            const plotTitle = data.is_portfolio ? 'Portfolio Simulation' : `${ticker.toUpperCase()} Stock Simulation`;
            plotData(data, plotTitle);

            if (data.is_portfolio && data.var_metrics) {
                document.getElementById('port-init').textContent = `$${data.var_metrics.initial_value.toFixed(2)}`;
                document.getElementById('port-var95').textContent = `-$${Math.abs(data.var_metrics.var_95).toFixed(2)}`;
                document.getElementById('port-var99').textContent = `-$${Math.abs(data.var_metrics.var_99).toFixed(2)}`;
                portfolioResults.style.display = 'block';
            }
            else if (data.options) {
                document.getElementById('call-price').textContent = `$${data.options.call_price.toFixed(2)}`;
                document.getElementById('put-price').textContent = `$${data.options.put_price.toFixed(2)}`;
                
                if(data.options.greeks) {
                    document.getElementById('delta-call').textContent = data.options.greeks.delta_call.toFixed(4);
                    document.getElementById('gamma').textContent = data.options.greeks.gamma.toFixed(4);
                    document.getElementById('theta-call').textContent = data.options.greeks.theta_call.toFixed(4);
                    document.getElementById('vega').textContent = data.options.greeks.vega.toFixed(4);
                }
                
                optionsResults.style.display = 'block';
            }

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // UI State: Reset
            btnText.style.display = 'block';
            loader.style.display = 'none';
            document.getElementById('simulate-btn').disabled = false;
        }
    });

    function plotData(data, title) {
        const traces = [];

        // Historical Data Trace
        traces.push({
            x: data.historical.dates,
            y: data.historical.prices,
            type: 'scatter',
            mode: 'lines',
            name: 'Historical',
            line: { color: '#3b82f6', width: 2 }
        });

        // Simulation Traces (limit to 20 for rendering speed if there are many)
        const simDates = data.simulation.dates;
        const simPaths = data.simulation.paths;
        
        const maxPathsToPlot = Math.min(simPaths.length, 20);

        for(let i=0; i<maxPathsToPlot; i++) {
            traces.push({
                x: simDates,
                y: simPaths[i],
                type: 'scatter',
                mode: 'lines',
                name: `Sim ${i + 1}`,
                line: { width: 1, opacity: 0.5 },
                hoverinfo: 'none' // Too noisy to hover on all simulations
            });
        }

        const layout = {
            title: title,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            xaxis: {
                title: 'Date',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Value ($)',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            showlegend: false,
            margin: { t: 50, r: 20, l: 50, b: 50 }
        };

        const config = { responsive: true };

        Plotly.newPlot('plot-container', traces, layout, config);
    }
});
