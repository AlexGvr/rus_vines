# google_mlkit_text_recognition ссылается на все скрипты; в APK подключён только latin.
-dontwarn com.google.mlkit.vision.text.chinese.**
-dontwarn com.google.mlkit.vision.text.devanagari.**
-dontwarn com.google.mlkit.vision.text.japanese.**
-dontwarn com.google.mlkit.vision.text.korean.**
# R8 стрипает внутренности ML Kit → NPE при создании распознавателя.
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.odml.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_text_common.** { *; }
