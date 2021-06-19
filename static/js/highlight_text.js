
function color_text(unis, uni_data){
    var i;
    for (i = 0; i < unis.length; i++) { 
        var hl_class = ("highlight" + String(i));
        var options = {
            "element": "mark",
            "className": hl_class,
            "accuracy": "exactly",
        }
        var context = document.querySelector("#highlight_text");
        var instance = new Mark(context);
        
        instance.mark(unis[i], options);

        console.log(uni_data[i])
        var alpha, styles;
        if (uni_data[i] > 0){
            alpha = uni_data[i] / 0.75;
            styles = {
            background: "rgba(255, 0, 0," + alpha + ')'
            //color: "rgba(255, 255, 255, 1.0)"
            };
        }
        else{
            alpha = -uni_data[i] / 0.75;
            styles = {
            background: "rgba(0, 0, 255," + alpha + ')'
            //color: "rgba(255, 255, 255, 1.0)"
            };
        }

        $(("mark."+hl_class)).css(styles);
    }
}

function color_texts(unis, uni_data){
    var i,j;
    
    color_strs = ['rgba(255, 0, 0,', 'rgba(0, 255, 0,', 'rgba(0, 0, 255,'];
    for(j = 0; j < uni_data.length; j++) { 
        for (i = 0; i < unis.length; i++) { 
            var hl_class = ("highlight" + String(j)+ String(i));
            var options = {
                "element": "mark",
                "className": hl_class,
                "accuracy": "partially",
            }
            var context = document.querySelector("#highlight_text");
            var instance = new Mark(context);
            
            instance.mark(unis[i], options);

            var alpha, styles;
            alpha = (0.0 -(uni_data[j][i]) - 1.0) / 5.0;
            styles = {
                background: color_strs[j] + alpha + ')'
            };

            $(("mark."+hl_class)).css(styles);
        }
    }
}