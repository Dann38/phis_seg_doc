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

var addSetClassify = function(id_image) {
    const xhr_set_classify = new XMLHttpRequest();
    xhr_set_classify.open("GET", "/get_set_classifier/"+id_image);
    xhr_set_classify.send();
    xhr_set_classify.onload = function() {
        if (xhr_set_classify.status == 200) {
            var response_set_classify = $.parseJSON(xhr_set_classify.response);
            $("#method-input")[0].value = response_set_classify.method;
            $("#coef-input")[0].value = response_set_classify.coef;
            $("#coef-input-text")[0].innerHTML =  response_set_classify.coef;
        }
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

var btnClassClick = function(e){
    d = new Date();
    $(".upload-file").attr("src", "/get_image_origin/"+e.target.dataset.rowId+"?"+d.getTime());
    $(".result-file").attr("src", "/get_image_result/"+e.target.dataset.rowId+"?"+d.getTime());
    $("#work-id").attr("data-row-id", e.target.dataset.rowId);
    addSetClassify(e.target.dataset.rowId)
}

$("#button_upload").click(function(){
    const fileInput = document.getElementById('file-input'); 
    
    const file = fileInput.files[0];

    const xhr = new XMLHttpRequest(); 
    const formData = new FormData();

    formData.append('file', file);

    xhr.open('POST', '/upload_image');
    xhr.send(formData);

    xhr.onload = function() {
        if (xhr.status != 200) { 
          alert(`Ошибка ${xhr.status}: ${xhr.statusText}`); 
        } else {
            var response_upload = $.parseJSON(xhr.response);
            addHistoryImg(response_upload.id_image, pr=true);
            $(".upload-file").attr("src", "/get_image_origin/"+response_upload.id_image);
            $(".result-file").attr("src", "/get_image_origin/"+response_upload.id_image);
            $("#work-id").attr("data-row-id", response_upload.id_image)
        }
      };
});


$("#button_classifier").click(function(){
    const id_image = $("#work-id")[0].dataset.rowId;

    const method_classifier = document.getElementById('method-input').value;
    const coef = document.getElementById('coef-input').value;
    console.log(id_image, method_classifier, coef)

    const xhr = new XMLHttpRequest();
    const formData = new FormData();


    formData.append('method_classifier', method_classifier);
    formData.append('coef', coef);

    xhr.open('POST', '/classify_image/'+id_image);
    xhr.send(formData);

    xhr.onload = function() {
        if (xhr.status != 200) {
          alert(`Ошибка ${xhr.status}: ${xhr.statusText}`);
        } else {
            d = new Date();
            $(".result-file").attr("src", "/get_image_result/"+id_image+"?"+d.getTime());
            addSetClassify(id_image);
        }
      };
});

addHistory();