function toDataURL(url, callback) {
  var xhr = new XMLHttpRequest();
  xhr.onload = function() {
    var reader = new FileReader();
    reader.onloadend = function() {
      callback(reader.result);
    }
    reader.readAsDataURL(xhr.response);
  };
  xhr.open('GET', url);
  xhr.responseType = 'blob';
  xhr.send();
}

toDataURL('https://www.gravatar.com/avatar/d50c83cc0c6523b4d3f6085295c953e0', function(dataUrl) {
  console.log('RESULT:', dataUrl)
})
