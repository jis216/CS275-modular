const colors = [Chart.helpers.color(window.chartColors.red), Chart.helpers.color(window.chartColors.blue), Chart.helpers.color(window.chartColors.green)]
function barPlot(canvas_id, set_names, data,labels){
    var fill_colors = data.map((value) => value > 0 ?  colors[0].alpha(0.5).rgbString() : colors[1].alpha(0.5).rgbString() );
    var line_colors = data.map((value) => value > 0 ?  colors[0].rgbString() : colors[1].rgbString() );
    var ctx = document.getElementById(canvas_id);
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: set_names,
                data: data,
                backgroundColor: fill_colors,
                borderColor: line_colors,
                borderWidth: 1
            }]
        },
        options: {
            legend: {
                display: false
            }
        }
        
    });
    return myChart;
};

function load_chart_data(labels, set_names, data){
    var color = Chart.helpers.color;
    var i;
    var data_sets = []
    for (i = 0; i < set_names.length; i++) { 
        data_sets.push({
            label: set_names[i],
            backgroundColor: colors[i].alpha(0.5).rgbString(),
            borderColor: colors[i],
            borderWidth: 1,
            data: data[i]
        })
    }

    var barChartData = {
        labels: labels,
        datasets: data_sets
    };

    return barChartData
}

function barPlots(set_names, data,labels) {
    var ctx = document.getElementById("bar_canvas").getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: load_chart_data(labels, set_names, data),
        options: {
            responsive: true,
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'word coefficients'
            }
        }
    });
};