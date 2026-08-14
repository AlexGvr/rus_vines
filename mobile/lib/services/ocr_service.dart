import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

abstract class OcrEngine {
  Future<String> recognizeFile(String path);

  void dispose();
}

/// ОГРАНИЧЕНИЕ: ML Kit Text Recognition v2 не имеет кириллической модели
/// (только latin/chinese/devanagari/japanese/korean). Латинские надписи
/// этикеток (бренды почти всегда дублируются латиницей) матчер закрывает
/// транслитерацией; чисто кириллические названия этим движком не читаются.
/// Прод-варианты для кириллицы: Apple Vision на iOS (ru поддерживается),
/// Tesseract через FFI, либо серверный fallback (Qwen2.5-VL).
/// Кроп/контраст-препроцессинг PWA сюда не переносим: это специфика
/// Tesseract, ML Kit сам детектит текстовые блоки на полном кадре.
class MlkitOcrEngine implements OcrEngine {
  MlkitOcrEngine()
      : _recognizer = TextRecognizer(script: TextRecognitionScript.latin);

  final TextRecognizer _recognizer;

  @override
  Future<String> recognizeFile(String path) async {
    final input = InputImage.fromFilePath(path);
    final recognized = await _recognizer.processImage(input);
    return recognized.text;
  }

  @override
  void dispose() {
    _recognizer.close();
  }
}
