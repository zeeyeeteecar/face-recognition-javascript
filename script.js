const video = document.getElementById("video");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceExpressionNet.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  //faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  //faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(startWebcam);

async function get_List_In_Labels_Folder() {
  const files = await fs.readdirSync("/labels");
  console.log("files", files);
  return files;
}

async function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
  console.log("get_List_In_Labels_Folder", await get_List_In_Labels_Folder());
}

function getLabeledFaceDescriptions() {
  const labels = ["TrumpD", "Obama", "TomT", "EllaH", "VincentK"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const photosFolder_1 = `./labels/${label}/${i}.png`;
        const photosFolder_2 = `http://www.accessrichmond.org/Captures/labels/${label}/${i}.png`;
        const photosFolder_3 =
          "https://www.accessrichmond.org/o2b2/CaptureCam/userPhotos/" +
          label +
          "/" +
          i.toString() +
          ".png";

        let headers = new Headers();

        headers.append("Content-Type", "application/json");
        headers.append("Accept", "application/json");
        //headers.append('Authorization', 'Basic ' + base64.encode(username + ":" +  password));
        headers.append("Origin", "https://www.accessrichmond.org/");

        const base64Response = await fetch(photosFolder_1, {
          mode: "no-cors",
        });

        const blob = await base64Response.blob();
        const img = await faceapi.bufferToImage(blob);

        // const img = await faceapi.fetchImage(photosFolder_2, {
        //   mode: "no-cors",
        //   //credentials: "include",
        //   method: "GET",
        //   headers: headers,
        // });

        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };

  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

    const results = resizedDetections.map((d) => {
      console.log("d.descriptor ==", d.descriptor);
      return faceMatcher.findBestMatch(d.descriptor);
    });

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      console.log("result", result);
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result,
      });
      drawBox.draw(canvas);
    });
  }, 1000);
});

//startWebcam();
