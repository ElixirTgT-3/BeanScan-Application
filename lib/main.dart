import 'package:flutter/material.dart';
import 'pages/history_page.dart';
import 'pages/scan_page.dart';
import 'pages/settings_page.dart';
import 'pages/splash_page.dart';
import 'utils/app_colors.dart';
import 'utils/app_constants.dart';
import 'utils/settings_service.dart';

void main() => runApp(const BeanScanApp());

class BeanScanApp extends StatefulWidget {
  const BeanScanApp({super.key});

  @override
  State<BeanScanApp> createState() => _BeanScanAppState();
}

class _BeanScanAppState extends State<BeanScanApp> {
  ThemeMode _themeMode = ThemeMode.system;

  @override
  void initState() {
    super.initState();
    _loadTheme();
  }

  Future<void> _loadTheme() async {
    final mode = await SettingsService.getThemeMode();
    setState(() {
      _themeMode = switch (mode) {
        AppThemeMode.light => ThemeMode.light,
        AppThemeMode.dark => ThemeMode.dark,
        AppThemeMode.system => ThemeMode.system,
      };
    });
  }

  void _onThemeChanged(AppThemeMode mode) async {
    await SettingsService.setThemeMode(mode);
    await _loadTheme();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: ThemeData(
        brightness: Brightness.light,
        scaffoldBackgroundColor: AppColors.lightBeige,
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.primaryBrown, brightness: Brightness.light),
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.primaryBrown, brightness: Brightness.dark),
      ),
      home: const SplashPage(),
      // Provide a route to main navigation with theme callback
      routes: {
        '/main': (_) => MainNavigationPage(onThemeChanged: _onThemeChanged),
      },
    );
  }
}

class MainNavigationPage extends StatefulWidget {
  final void Function(AppThemeMode mode)? onThemeChanged;
  const MainNavigationPage({super.key, this.onThemeChanged});

  @override
  State<MainNavigationPage> createState() => _MainNavigationPageState();
}

class _MainNavigationPageState extends State<MainNavigationPage> {
  int currentPageIndex = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      const HistoryPage(),
      ScanPage(
        onClose: () => setState(() => currentPageIndex = 0),
      ),
      const SettingsPage(),
    ];

    return Scaffold(
      body: pages[currentPageIndex],
      bottomNavigationBar: currentPageIndex == 1 
          ? null 
          : _buildBottomNavigationBar(),
      floatingActionButton: currentPageIndex == 1 
          ? null 
          : _buildScanButton(),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }

  Widget _buildBottomNavigationBar() {
    return Container(
      color: Colors.white,
      child: SafeArea(
        child: SizedBox(
          height: AppConstants.navigationBarHeight,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              IconButton(
                icon: Icon(
                  Icons.history,
                  color: AppColors.primaryBrown,
                  size: currentPageIndex == 0 ? 28 : 24,
                ),
                onPressed: () => setState(() => currentPageIndex = 0),
              ),
              SizedBox(width: AppConstants.centerButtonSpace),
              IconButton(
                icon: Icon(
                  Icons.settings,
                  color: AppColors.primaryBrown,
                  size: currentPageIndex == 2 ? 28 : 24,
                ),
                onPressed: () => setState(() => currentPageIndex = 2),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScanButton() {
    return FloatingActionButton(
      backgroundColor: AppColors.primaryBrown,
      shape: const CircleBorder(),
      child: const Icon(Icons.qr_code_scanner, color: Colors.white),
      onPressed: () => setState(() => currentPageIndex = 1),
    );
  }
}
