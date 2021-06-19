
$(document).ready(function() {
    $('form').on('submit', function(event) {
        $.ajax({
            data : {
                test_text : $('#test_text').val(),
                },
            type : 'POST',
            url : '/process'
        })
        .always(function(data) {
            console.log(JSON.stringify(data));
        })
        .done(function(data) {
            alert("Success.")
            /** 
            console.log(data.confidence)
            $('#output').text(data.output).show();
            $('#confidence').text(data.confidence).show();
            alert("Success.");
            */
        });
     });
});
/**
$(function(){
	$('button').click(function(){
		var user = $('#txtUsername').val();
		var pass = $('#txtPassword').val();
		$.ajax({
			url: '/process',
			data: { test_text : $('#test_text').val()},
            type: 'POST',
            async:false,
			success: function(response){
                alert('success');
				console.log(response);
			},
			error: function(error){
                alert('fail');
				console.log(error);
			}
		});
	});
});
*/