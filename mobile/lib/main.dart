import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'screens/catalog_screen.dart';
import 'screens/my_wines_screen.dart';
import 'screens/scan_screen.dart';
import 'services/my_wines_store.dart';
import 'services/wine_repository.dart';
import 'widgets/wine_widgets.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SvoeVinoApp());
}

class SvoeVinoApp extends StatelessWidget {
  const SvoeVinoApp({super.key});

  ThemeData _theme(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(
      seedColor: kWineColor,
      brightness: brightness,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      brightness: brightness,
      appBarTheme: const AppBarTheme(centerTitle: false),
      cardTheme: const CardThemeData(elevation: 1),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Своё вино',
      debugShowCheckedModeBanner: false,
      locale: const Locale('ru'),
      supportedLocales: const [Locale('ru'), Locale('en')],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      theme: _theme(Brightness.light),
      darkTheme: _theme(Brightness.dark),
      themeMode: ThemeMode.system,
      home: const _AppLoader(),
    );
  }
}

/// Загружает репозиторий вин и локальное хранилище, показывая сплэш.
class _AppLoader extends StatefulWidget {
  const _AppLoader();

  @override
  State<_AppLoader> createState() => _AppLoaderState();
}

class _AppLoaderState extends State<_AppLoader> {
  late Future<(WineRepository, MyWinesStore)> _future;

  @override
  void initState() {
    super.initState();
    _future = _load();
  }

  Future<(WineRepository, MyWinesStore)> _load() async {
    final repo = await WineRepository.load();
    final store = await MyWinesStore.load();
    return (repo, store);
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<(WineRepository, MyWinesStore)>(
      future: _future,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Scaffold(
            body: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text('🍷', style: TextStyle(fontSize: 48)),
                    const SizedBox(height: 12),
                    const Text('Не удалось загрузить каталог вин'),
                    const SizedBox(height: 12),
                    FilledButton(
                      onPressed: () => setState(() => _future = _load()),
                      child: const Text('Повторить'),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
        if (!snapshot.hasData) {
          return const _SplashScreen();
        }
        final (repo, store) = snapshot.data!;
        return HomeShell(repository: repo, store: store);
      },
    );
  }
}

class _SplashScreen extends StatelessWidget {
  const _SplashScreen();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF7B1E3A), Color(0xFF3E0F1E)],
          ),
        ),
        child: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('🍷', style: TextStyle(fontSize: 64)),
              SizedBox(height: 16),
              Text(
                'Своё вино',
                style: TextStyle(
                    color: Colors.white, fontSize: 28, fontWeight: FontWeight.w700),
              ),
              SizedBox(height: 8),
              Text(
                'Российские вина — сканируй и выбирай',
                style: TextStyle(color: Colors.white70, fontSize: 14),
              ),
              SizedBox(height: 24),
              SizedBox(
                width: 28,
                height: 28,
                child: CircularProgressIndicator(color: kGoldColor, strokeWidth: 3),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Корневой каркас с тремя вкладками.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.repository, required this.store});

  final WineRepository repository;
  final MyWinesStore store;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;

  /// Счётчик для принудительного пересоздания вкладки «Мои вина»,
  /// чтобы она перечитывала store при каждом открытии.
  int _myWinesRefresh = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: IndexedStack(
          index: _tab,
          children: [
            ScanScreen(repository: widget.repository, store: widget.store),
            CatalogScreen(repository: widget.repository, store: widget.store),
            MyWinesScreen(
              key: ValueKey('my-wines-$_myWinesRefresh'),
              repository: widget.repository,
              store: widget.store,
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) {
          setState(() {
            _tab = i;
            if (i == 2) _myWinesRefresh++;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.photo_camera_outlined),
            selectedIcon: Icon(Icons.photo_camera),
            label: 'Скан',
          ),
          NavigationDestination(
            icon: Icon(Icons.wine_bar_outlined),
            selectedIcon: Icon(Icons.wine_bar),
            label: 'Каталог',
          ),
          NavigationDestination(
            icon: Icon(Icons.favorite_outline),
            selectedIcon: Icon(Icons.favorite),
            label: 'Мои вина',
          ),
        ],
      ),
    );
  }
}
