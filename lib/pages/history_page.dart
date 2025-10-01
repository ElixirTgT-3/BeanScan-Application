import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../utils/app_constants.dart';
import '../utils/api_service.dart';
import 'results_page.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<dynamic> _items = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final res = await ApiService.fetchHistory(limit: 50);
    if (!mounted) return;
    setState(() {
      _loading = false;
      if (res['success'] == true) {
        _items = (res['data']['scans'] as List?) ?? [];
      }
    });
  }

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
            "History",
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
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : (_items.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: const [
                        Text(
                          "You don't have any history",
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: AppColors.primaryBrown,
                          ),
                        ),
                        SizedBox(height: AppConstants.smallSpacing),
                        Text(
                          "Once you scan a coffee bean, results will show up here.",
                          style: TextStyle(fontSize: 12, color: AppColors.textGrey),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.all(AppConstants.defaultPadding),
                    itemBuilder: (_, index) => _historyTile(_items[index]),
                    separatorBuilder: (_, __) => const SizedBox(height: AppConstants.smallSpacing),
                    itemCount: _items.length,
                  )),
      ),
    );
  }

  Widget _historyTile(dynamic item) {
    final createdAt = item['created_at'] ?? '';
    final beanType = item['bean_type_name'] ?? item['bean_type'] ?? 'Unknown';
    final healthy = item['healthy_percent'] ?? 0;
    final defective = item['defective_percent'] ?? 0;

    return InkWell(
      onTap: () async {
        // Fetch full details then navigate to ResultsPage-like detail
        final details = await ApiService.fetchHistoryDetails(item['history_id']);
        if (!mounted || details['success'] != true) return;
        final data = details['data'];

        // Build prediction from details to match live scan shape
        final prediction = BeanPrediction(
          prediction: data['bean_type']?['type_name'] ?? beanType,
          confidence: (data['shelf_life']?['confidence_score'] ?? item['confidence_score'] ?? healthy / 100).toDouble(),
          allProbabilities: const {},
        );

        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => ResultsPage(
              prediction: prediction,
              imagePath: (data['image']?['image_url'] ?? '') as String,
              defectDetection: data['defect_detection'],
              shelfLife: data['shelf_life'],
            ),
          ),
        );
      },
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(AppConstants.mediumRadius),
          border: Border.all(color: AppColors.dividerGrey, width: AppConstants.thinBorder),
        ),
        padding: const EdgeInsets.all(AppConstants.defaultPadding),
        child: Row(
          children: [
            Container(
              width: AppConstants.iconButtonSize,
              height: AppConstants.iconButtonSize,
              decoration: BoxDecoration(
                color: AppColors.iconBackground,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(Icons.coffee, color: AppColors.primaryBrown, size: AppConstants.smallIconSize),
            ),
            const SizedBox(width: AppConstants.mediumSpacing),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _formatDate(createdAt),
                    style: const TextStyle(color: AppColors.textDarkGrey, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Text('Type: $beanType', style: const TextStyle(color: AppColors.textDarkGrey, fontSize: 12)),
                  const SizedBox(height: 2),
                  Text('Healthy: ${healthy.toString()}%  |  Defects: ${defective.toString()}%', style: const TextStyle(color: AppColors.textGrey, fontSize: 11)),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: AppColors.primaryBrown),
          ],
        ),
      ),
    );
  }

  String _formatDate(String iso) {
    try {
      final dt = DateTime.tryParse(iso)?.toLocal();
      if (dt == null) return iso;
      final mm = dt.month.toString().padLeft(2, '0');
      final dd = dt.day.toString().padLeft(2, '0');
      final yyyy = dt.year.toString();
      final hh = dt.hour.toString().padLeft(2, '0');
      final min = dt.minute.toString().padLeft(2, '0');
      return '$mm/$dd/$yyyy • $hh:$min';
    } catch (_) {
      return iso;
    }
  }
} 