import 'package:flutter/material.dart';
import 'pages/history_page.dart';
import 'pages/scan_page.dart';
import 'pages/settings_page.dart';
import 'pages/splash_page.dart';
import 'utils/app_colors.dart';
import 'utils/app_constants.dart';
import 'utils/app_settings.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppSettings.instance.load();
  runApp(const BeanScanApp());
}

class BeanScanApp extends StatelessWidget {
  const BeanScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: AppSettings.instance,
      builder: (context, _) {
        final settings = AppSettings.instance;
        const lightBackground = AppColors.lightBeige;
        final lightColorScheme = ColorScheme.fromSeed(
          seedColor: AppColors.primaryBrown,
        ).copyWith(
          primary: AppColors.primaryBrown,
          onPrimary: Colors.white,
          secondary: const Color(0xFFB08A58),
          onSecondary: Colors.white,
          surface: Colors.white,
          surfaceContainerHighest: const Color(0xFFF3E9DC),
          onSurface: AppColors.textDarkGrey,
          onSurfaceVariant: AppColors.textGrey,
          secondaryContainer: const Color(0xFFE8D8C4),
          onSecondaryContainer: AppColors.textDarkGrey,
          outline: AppColors.dividerGrey,
          surfaceTint: AppColors.primaryBrown,
        );
        const darkBackground = Color(0xFF1F1F1F);
        final darkColorScheme = ColorScheme.fromSeed(
          seedColor: AppColors.primaryBrown,
          brightness: Brightness.dark,
        ).copyWith(
          primary: AppColors.primaryBrown,
          onPrimary: Colors.white,
          surface: const Color(0xFF2A2420),
          surfaceContainerHighest: const Color(0xFF332B26),
          onSurface: const Color(0xFFE5D6C2),
          onSurfaceVariant: const Color(0xFFD9C7B0),
          secondary: const Color(0xFFB08A58),
          onSecondary: Colors.black,
          secondaryContainer: const Color(0xFF3A2A1F),
          onSecondaryContainer: const Color(0xFFE8D4B5),
          tertiary: const Color(0xFF7A5A33),
          onTertiary: Colors.white,
          outline: const Color(0xFF5A4A3F),
          outlineVariant: const Color(0xFF3E332C),
          surfaceTint: const Color(0xFF3A302A),
          inverseSurface: const Color(0xFFE5D6C2),
          onInverseSurface: const Color(0xFF1A120D),
          inversePrimary: const Color(0xFFD4A373),
        );

        return MaterialApp(
          debugShowCheckedModeBanner: false,
          themeMode: settings.themeMode,
          theme: ThemeData(
            colorScheme: lightColorScheme,
            scaffoldBackgroundColor: lightBackground,
            appBarTheme: AppBarTheme(
              backgroundColor: lightBackground,
              foregroundColor: lightColorScheme.primary,
              elevation: 0,
            ),
          ),
          darkTheme: ThemeData(
            brightness: Brightness.dark,
            colorScheme: darkColorScheme,
            scaffoldBackgroundColor: darkBackground,
            appBarTheme: AppBarTheme(
              backgroundColor: darkBackground,
              foregroundColor: darkColorScheme.primary,
              elevation: 0,
            ),
            cardColor: darkColorScheme.surface,
            dividerColor: darkColorScheme.outline,
            iconTheme: IconThemeData(color: darkColorScheme.primary),
            textTheme: ThemeData(brightness: Brightness.dark).textTheme.apply(
                  bodyColor: darkColorScheme.onSurface,
                  displayColor: darkColorScheme.onSurface,
                ),
            filledButtonTheme: FilledButtonThemeData(
              style: FilledButton.styleFrom(
                backgroundColor: darkColorScheme.primary,
                foregroundColor: darkColorScheme.onPrimary,
                shape: const StadiumBorder(),
              ),
            ),
          ),
          home: const SplashPage(),
        );
      },
    );
  }
}

class MainNavigationPage extends StatefulWidget {
  const MainNavigationPage({super.key});

  @override
  State<MainNavigationPage> createState() => _MainNavigationPageState();
}

class _MainNavigationPageState extends State<MainNavigationPage> {
  final HistoryPageController _historyController = HistoryPageController();
  int currentPageIndex = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      HistoryPage(controller: _historyController),
      ScanPage(
        onClose: () {
          setState(() => currentPageIndex = 0);
          _historyController.refresh(showLoadingIndicator: true);
        },
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
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final activeColor = colorScheme.primary;
    final inactiveColor = colorScheme.onSurfaceVariant;
    return SafeArea(
      top: false,
      child: Container(
        height: AppConstants.navigationBarHeight,
        padding: const EdgeInsets.symmetric(horizontal: AppConstants.largePadding),
        decoration: BoxDecoration(
          color: colorScheme.surface,
          border: Border(
            top: BorderSide(
              color: colorScheme.outline.withValues(alpha: 0.2),
              width: 1,
            ),
          ),
        ),
        child: Row(
          children: [
            Padding(
              padding: const EdgeInsets.only(right: AppConstants.smallSpacing),
              child: IconButton(
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(minWidth: 44, minHeight: 44),
                icon: Icon(
                  Icons.history,
                  color: currentPageIndex == 0 ? activeColor : inactiveColor,
                  size: currentPageIndex == 0 ? 28 : 24,
                ),
                onPressed: () {
                  if (currentPageIndex != 0) {
                    setState(() => currentPageIndex = 0);
                  }
                  _historyController.refresh();
                },
              ),
            ),
            const Spacer(),
            SizedBox(width: AppConstants.centerButtonSpace),
            const Spacer(),
            Padding(
              padding: const EdgeInsets.only(left: AppConstants.smallSpacing),
              child: IconButton(
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(minWidth: 44, minHeight: 44),
                icon: Icon(
                  Icons.settings,
                  color: currentPageIndex == 2 ? activeColor : inactiveColor,
                  size: currentPageIndex == 2 ? 28 : 24,
                ),
                onPressed: () => setState(() => currentPageIndex = 2),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildScanButton() {
    return FloatingActionButton(
      backgroundColor: Theme.of(context).colorScheme.primary,
      shape: const CircleBorder(),
      child: Image.asset(
        'assets/images/icons/scan.png',
        width: 24,
        height: 24,
        color: Theme.of(context).colorScheme.onPrimary,
      ),
      onPressed: () => setState(() => currentPageIndex = 1),
    );
  }
}
