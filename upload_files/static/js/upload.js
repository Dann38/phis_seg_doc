var addHistoryImg = function(id, pr=false) {
    let new_history_img = `<div class="card" style="width: 100%;">
             <img src="/get_image_origin/`+id+ `" class="card-img-top" alt="...">
             <div class="card-body">
                  <a href="#" class="button_id_send btn btn-primary" data-row-id="`+id +`">id: `+id+`</a>
             </div>
        </div>`;
    if (pr) {
        $("#history-group").prepend(new_history_img)
    }else{
        $("#history-group").append(new_history_img);
    }

    $(".button_id_send").on("click", btnClassClick);
 }

 var addHistory = function() {
    const xhr_history = new XMLHttpRequest();
    xhr_history.open("GET", "/get_history");
    xhr_history.send();
    xhr_history.onload = function() {
        if (xhr_history.status == 200) {
            var response_history = $.parseJSON(xhr_history.response);
            var array = response_history.id_list;
            for(var i=0; i < array.length ; i++){
                addHistoryImg(array[i]);
            }
        }
    }
    $(".button_id_send").on("click", btnClassClick);
 }
addHistory();

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
            addHistoryImg(response_upload.id_image, pr=true);
            $(".upload-file").attr("src", "/get_image_origin/"+response_upload.id_image);
            $(".result-file").attr("src", "/get_image_result/"+response_upload.id_image);
        }
      };
});

var btnClassClick = function(e){
    $(".upload-file").attr("src", "/get_image_origin/"+e.target.dataset.rowId);
    $(".result-file").attr("src", "/get_image_result/"+e.target.dataset.rowId);
}

