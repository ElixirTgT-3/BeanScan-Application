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
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          themeMode: settings.themeMode,
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: AppColors.primaryBrown,
              background: AppColors.lightBeige,
            ),
            scaffoldBackgroundColor: AppColors.lightBeige,
            appBarTheme: const AppBarTheme(
              backgroundColor: AppColors.lightBeige,
              foregroundColor: AppColors.primaryBrown,
              elevation: 0,
            ),
          ),
          darkTheme: ThemeData(
            brightness: Brightness.dark,
            colorScheme: ColorScheme.fromSeed(
              seedColor: AppColors.primaryBrown,
              brightness: Brightness.dark,
            ),
            scaffoldBackgroundColor: Colors.grey[900],
            appBarTheme: AppBarTheme(
              backgroundColor: Colors.grey[900],
              foregroundColor: Colors.white,
              elevation: 0,
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
                onPressed: () {
                  if (currentPageIndex != 0) {
                    setState(() => currentPageIndex = 0);
                  }
                  _historyController.refresh();
                },
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
      child: Image.asset(
        'assets/images/icons/scan.png',
        width: 24,
        height: 24,
        color: Colors.white,
      ),
      onPressed: () => setState(() => currentPageIndex = 1),
    );
  }
}
