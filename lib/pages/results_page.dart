import 'package:flutter/material.dart';
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
          'Scan Results',
          style: TextStyle(color: Colors.white),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(AppConstants.largePadding),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildPredictionCard(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildConfidenceBar(),
              const SizedBox(height: AppConstants.largeSpacing),
              _buildAllProbabilities(),
              const Spacer(),
              _buildActionButtons(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return const Text(
      'Bean Type Analysis',
      style: TextStyle(
        fontSize: 24,
        fontWeight: FontWeight.bold,
        color: Colors.white,
      ),
    );
  }

  Widget _buildPredictionCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        children: [
          const Icon(
            Icons.coffee,
            size: 48,
            color: AppColors.primaryBrown,
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          Text(
            prediction.prediction,
            style: const TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: AppColors.textDarkGrey,
            ),
          ),
          const SizedBox(height: AppConstants.smallSpacing),
          Text(
            'Detected Bean Type',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[600],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfidenceBar() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Confidence Level',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.textDarkGrey,
            ),
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          LinearProgressIndicator(
            value: prediction.confidence,
            backgroundColor: Colors.grey[300],
            valueColor: AlwaysStoppedAnimation<Color>(
              prediction.confidence > 0.8 ? Colors.green : 
              prediction.confidence > 0.6 ? Colors.orange : Colors.red,
            ),
            minHeight: 8,
          ),
          const SizedBox(height: AppConstants.smallSpacing),
          Text(
            '${(prediction.confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: prediction.confidence > 0.8 ? Colors.green : 
                     prediction.confidence > 0.6 ? Colors.orange : Colors.red,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAllProbabilities() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(AppConstants.largePadding),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'All Bean Types',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.textDarkGrey,
            ),
          ),
          const SizedBox(height: AppConstants.mediumSpacing),
          ...prediction.allProbabilities.entries.map((entry) {
            final isTopPrediction = entry.key == prediction.prediction;
            return Padding(
              padding: const EdgeInsets.only(bottom: AppConstants.smallSpacing),
              child: Row(
                children: [
                  Expanded(
                    flex: 2,
                    child: Text(
                      entry.key,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: isTopPrediction ? FontWeight.w600 : FontWeight.normal,
                        color: isTopPrediction ? AppColors.primaryBrown : AppColors.textDarkGrey,
                      ),
                    ),
                  ),
                  Expanded(
                    flex: 3,
                    child: LinearProgressIndicator(
                      value: entry.value,
                      backgroundColor: Colors.grey[200],
                      valueColor: AlwaysStoppedAnimation<Color>(
                        isTopPrediction ? AppColors.primaryBrown : Colors.grey[400]!,
                      ),
                      minHeight: 6,
                    ),
                  ),
                  const SizedBox(width: AppConstants.smallSpacing),
                  SizedBox(
                    width: 50,
                    child: Text(
                      '${(entry.value * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: isTopPrediction ? FontWeight.w600 : FontWeight.normal,
                        color: isTopPrediction ? AppColors.primaryBrown : Colors.grey[600],
                      ),
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey[300],
              foregroundColor: AppColors.textDarkGrey,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('Scan Again'),
          ),
        ),
        const SizedBox(width: AppConstants.mediumSpacing),
        Expanded(
          child: ElevatedButton(
            onPressed: () {
              // TODO: Save to history
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Saved to history'),
                  backgroundColor: Colors.green,
                ),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBrown,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: AppConstants.mediumSpacing),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
              ),
            ),
            child: const Text('Save Result'),
          ),
        ),
      ],
    );
  }
}
