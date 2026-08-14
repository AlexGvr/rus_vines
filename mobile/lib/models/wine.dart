class Wine {
  final String slug;
  final String title;
  final String manufacturer;
  final String region;
  final String category;
  final String color;
  final String sweetness;
  final String wineColor;
  final String temperature;
  final String description;
  final double? alcohol;
  final double? rating;
  final List<String> grapes;
  final List<String> dishes;
  final String? photo;

  Wine({
    required this.slug,
    required this.title,
    required this.manufacturer,
    required this.region,
    required this.category,
    required this.color,
    required this.sweetness,
    required this.wineColor,
    required this.temperature,
    required this.description,
    required this.alcohol,
    required this.rating,
    required this.grapes,
    required this.dishes,
    required this.photo,
  });

  factory Wine.fromJson(Map<String, dynamic> json) {
    return Wine(
      slug: (json['slug'] as String?) ?? '',
      title: (json['title'] as String?) ?? '',
      manufacturer: (json['manufacturer'] as String?) ?? '',
      region: (json['region'] as String?) ?? '',
      category: (json['category'] as String?) ?? '',
      color: (json['color'] as String?) ?? '',
      sweetness: (json['sweetness'] as String?) ?? '',
      wineColor: (json['wineColor'] as String?) ?? '',
      temperature: (json['temperature'] as String?) ?? '',
      description: (json['description'] as String?) ?? '',
      alcohol: (json['alcohol'] as num?)?.toDouble(),
      rating: (json['rating'] as num?)?.toDouble(),
      grapes: ((json['grapes'] as List?) ?? const [])
          .map((e) => e as String)
          .toList(),
      dishes: ((json['dishes'] as List?) ?? const [])
          .map((e) => e as String)
          .toList(),
      photo: json['photo'] as String?,
    );
  }
}
