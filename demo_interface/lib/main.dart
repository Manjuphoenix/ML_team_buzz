import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
//try wrapping in material app?
  Widget build(BuildContext context) {
    return const MaterialApp(home: CameraWidgetState());
  }
}

class CameraWidgetState extends StatefulWidget {
  const CameraWidgetState({Key? key}) : super(key: key);

  @override
  State<CameraWidgetState> createState() => _CameraWidgetState();
}

class _CameraWidgetState extends State<CameraWidgetState> {
  PickedFile? imageFile;

  Future<void> _showChoiceDialog(BuildContext context) {
    return showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            content: SingleChildScrollView(
              child: ListBody(
                children: [
                  const Divider(
                    height: 1,
                    color: Color.fromARGB(128, 128, 128, 10),
                  ),
                  ListTile(
                    onTap: () {
                      _openGallery(context);
                    },
                    title: const Text("Gallery"),
                    leading: const Icon(
                      Icons.account_box,
                      color: Color.fromARGB(128, 128, 128, 10),
                    ),
                  ),
                  const Divider(
                    height: 1,
                    color: Color.fromARGB(128, 128, 128, 10),
                  ),
                  ListTile(
                    onTap: () {
                      _openCamera(context);
                    },
                    title: const Text("Camera"),
                    leading: const Icon(
                      Icons.camera,
                      color: Color.fromARGB(128, 128, 128, 10),
                    ),
                  ),
                ],
              ),
            ),
          );
        });
  }

  @override
  Widget build(BuildContext context) {
    // Implement build
    return MaterialApp(
        home: Scaffold(
      appBar: AppBar(
        title: const Text("Pick Image Camera"),
        backgroundColor: const Color.fromRGBO(128, 128, 0, 10),
      ),
      body: Center(
        child: Container(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Card(
                child: (imageFile == null)
                    ? const Text("Choose Image")
                    : Image.file(File(imageFile!.path)),
              ),
              MaterialButton(
                textColor: Colors.white,
                color: Color.fromRGBO(128, 128, 0, 10),
                onPressed: () {
                  _showChoiceDialog(context);
                },
                child: const Text("Select Image"),
              )
            ],
          ),
        ),
      ),
    ));
  }

  void _openGallery(BuildContext context) async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.gallery,
    );
    setState(() {
      imageFile = PickedFile(pickedFile!.path);
    });

    Navigator.pop(context);
  }

  void _openCamera(BuildContext context) async {
    final pickedFile = await ImagePicker().pickImage(
      source: ImageSource.camera,
    );
    setState(() {
      imageFile = PickedFile(pickedFile!.path);
    });
    Navigator.pop(context);
  }
}
