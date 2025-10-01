import 'package:flutter/material.dart';
import 'pages/history_page.dart';
import 'pages/scan_page.dart';
import 'pages/settings_page.dart';
import 'pages/splash_page.dart';
import 'utils/app_colors.dart';
import 'utils/app_constants.dart';

void main() => runApp(const BeanScanApp());

class BeanScanApp extends StatelessWidget {
  const BeanScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.lightBeige,
      ),
      home: const SplashPage(),
    );
  }
}

class MainNavigationPage extends StatefulWidget {
  const MainNavigationPage({super.key});

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
