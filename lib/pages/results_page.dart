import 'package:flutter/material.dart';
import 'dart:io';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';

class ResultsPage extends StatelessWidget {
  final BeanPrediction prediction;
  final String imagePath;

  const ResultsPage({
    super.key,
    required this.prediction,
    required this.imagePath,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.scanDarkGrey,
      appBar: AppBar(
        backgroundColor: AppColors.scanDarkGrey,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Scanned Coffee Bean Result',
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.largePadding),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildImagePreview(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildInfoCard(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildHealthTiles(),
              const Spacer(),
              _buildYesNoButtons(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      height: 210,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.15),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
        border: Border.all(color: Colors.white24),
      ),
      clipBehavior: Clip.antiAlias,
      child: imagePath.isNotEmpty
          ? Image.file(File(imagePath), fit: BoxFit.cover)
          : const Center(
              child: Icon(Icons.image, color: Colors.white54, size: 48),
            ),
    );
  }

  Widget _buildInfoCard() {
    final DateTime now = DateTime.now();
    final String dateStr = '${now.month}/${now.day}/${now.year} - ${now.hour}:${now.minute.toString().padLeft(2, '0')}';
    final double healthyPct = (prediction.confidence * 100).clamp(0, 100);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(dateStr, style: const TextStyle(color: AppColors.textDarkGrey)),
              const Icon(Icons.refresh, size: 18, color: AppColors.textDarkGrey),
            ],
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          Text('Type: ${prediction.prediction}', style: const TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 4),
          const Text('Mold: - | Bleached: -', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 4),
          const Text('Total Beans: -', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: AppConstants.mediumSpacing),
          const Text('Estimated Shelf Life', style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.textDarkGrey)),
          const SizedBox(height: 2),
          const Text('Confidence Score:', style: TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: 6),
          Row(
            children: [
              _circularScore(label: 'Confidence', percent: healthyPct, color: AppColors.primaryBrown),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHealthTiles() {
    final double healthy = (prediction.confidence * 100).clamp(0, 100);
    final double defective = (100 - healthy).clamp(0, 100);
    return Row(
      children: [
        Expanded(child: _healthCard('Healthy:', healthy, Colors.green)),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(child: _healthCard('Defective:', defective, AppColors.primaryBrown)),
      ],
    );
  }

  Widget _healthCard(String title, double percent, Color color) {
    return Container(
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(color: AppColors.textDarkGrey)),
          const SizedBox(height: AppConstants.smallSpacing),
          Center(child: _circularPercent(percent: percent, color: color)),
        ],
      ),
    );
  }

  Widget _circularPercent({required double percent, required Color color}) {
    return SizedBox(
      height: 90,
      width: 90,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: percent / 100.0,
            strokeWidth: 8,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
          Text('${percent.toStringAsFixed(0)}%', style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _circularScore({required String label, required double percent, required Color color}) {
    return Row(
      children: [
        _circularPercent(percent: percent, color: color),
        const SizedBox(width: AppConstants.mediumSpacing),
        Text(label, style: const TextStyle(color: AppColors.textDarkGrey)),
      ],
    );
  }

  Widget _buildYesNoButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBrown,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('Yes'),
          ),
        ),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).maybePop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[300],
              foregroundColor: AppColors.textDarkGrey,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('No'),
          ),
        ),
      ],
    );
  }
}
