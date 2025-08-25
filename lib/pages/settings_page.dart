import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _buildHeader(),
        _buildContent(),
      ],
    );
  }

  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppColors.headerGrey,
      ),
      child: SafeArea(
        bottom: false,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: AppConstants.headerPadding),
          child: const Text(
            "Settings",
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: AppColors.primaryBrown,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    return Expanded(
      child: Container(
        color: AppColors.headerGrey,
        child: ListView(
          padding: const EdgeInsets.all(AppConstants.defaultPadding),
          children: [
            // Scanning Preferences Section
            _buildSectionTitle("Scanning Preferences"),
            _buildSettingItem(Icons.crop_square, "Default Scan Mode"),
            _buildSettingItem(Icons.camera_alt, "Camera Settings"),
            _buildSettingItem(Icons.landscape, "Image Quality"),
            _buildSettingItem(Icons.download, "Auto-Save Scans"),
            
            const SizedBox(height: AppConstants.extraLargePadding),
            
            // App Preferences Section
            _buildSectionTitle("App Preferences"),
            _buildSettingItem(Icons.language, "Language"),
            _buildSettingItem(Icons.dark_mode, "Theme"),
            
            const SizedBox(height: AppConstants.extraLargePadding),
            
            // Support & About Section
            _buildSectionTitle("Support & About"),
            _buildSettingItem(Icons.help, "Help Center"),
            _buildSettingItem(Icons.info, "About the App"),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: AppConstants.mediumSpacing, top: AppConstants.smallSpacing),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
          color: AppColors.primaryBrown,
        ),
      ),
    );
  }

  Widget _buildSettingItem(IconData icon, String title) {
    return Container(
      margin: const EdgeInsets.only(bottom: 1),
      decoration: const BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: AppColors.dividerGrey,
            width: AppConstants.thinBorder,
          ),
        ),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(vertical: AppConstants.smallSpacing, horizontal: 0),
        leading: Container(
          width: AppConstants.iconButtonSize,
          height: AppConstants.iconButtonSize,
          decoration: BoxDecoration(
            color: AppColors.iconBackground,
            borderRadius: BorderRadius.circular(AppConstants.largeRadius),
          ),
          child: Icon(
            icon,
            color: AppColors.primaryBrown,
            size: AppConstants.smallIconSize,
          ),
        ),
        title: Text(
          title,
          style: const TextStyle(
            color: AppColors.primaryBrown,
            fontSize: 16,
          ),
        ),
        trailing: const Icon(
          Icons.chevron_right,
          color: AppColors.primaryBrown,
        ),
      ),
    );
  }
} 