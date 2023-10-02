import { Component } from '@angular/core';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent {
  text_block:string= "Область загрузки файла";
  is_upload = false
  ngOnInit(){

  }
  upload(){
  if (!this.is_upload){
    this.text_block = "Загружено";
    this.is_upload = true
  }else{
    this.text_block= "Область загрузки файла";
    this.is_upload = false
  }

  }
}
