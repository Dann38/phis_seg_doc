$("#button_send").click(function(){
    const fileInput = document.getElementById('file-input'); 
    
    const file = fileInput.files[0]; 

    const xhr = new XMLHttpRequest(); 
    const formData = new FormData(); 

    formData.append('file', file); 

    xhr.open('POST', '/upload');
    xhr.send(formData);

    xhr.onload = function() {
        if (xhr.status != 200) { 
          alert(`Ошибка ${xhr.status}: ${xhr.statusText}`); 
        } else {
            var response_upload = $.parseJSON(xhr.response);
            $(".upload-file").attr("src", "/get_image_origin/"+response_upload.id_image);
            $(".result-file").attr("src", "/get_image_result/"+response_upload.id_image);
        }
      };
});

var btnClassClick = function(e){
    $(".upload-file").attr("src", "/get_image_origin/"+e.target.dataset.rowId);
    $(".result-file").attr("src", "/get_image_result/"+e.target.dataset.rowId);
}
$(".button_id_send").on("click", btnClassClick);