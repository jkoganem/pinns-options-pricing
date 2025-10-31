#!/bin/bash
# Morning Status Check Script
# Run this when you wake up to see overnight results

echo "================================================================================"
echo "GOOD MORNING! OVERNIGHT STATUS CHECK"
echo "================================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "================================================================================"
echo "1. OPTUNA HYPERPARAMETER TUNING STATUS"
echo "================================================================================"

# Check if Optuna log exists
if [ -f "optuna_tuning.log" ]; then
    echo "✓ Optuna log found"
    echo ""
    echo "Last 30 lines of output:"
    echo "--------------------------------------------------------------------------------"
    tail -30 optuna_tuning.log
    echo ""

    # Check if completed
    if grep -q "OPTIMIZATION COMPLETE" optuna_tuning.log; then
        echo "✅ STATUS: OPTUNA TUNING COMPLETED!"
        echo ""

        # Show best result
        if grep -q "Best mean error" optuna_tuning.log; then
            echo "Best results:"
            grep "Best mean error" optuna_tuning.log
            grep "Best max error" optuna_tuning.log
            grep "Training time" optuna_tuning.log | head -1
        fi
        echo ""

        # Check if results file exists
        if [ -f "hyperparameter_tuning_optuna_results.json" ]; then
            echo "✓ Results saved to: hyperparameter_tuning_optuna_results.json"
        fi

        # Check if figures exist
        echo ""
        echo "Generated figures:"
        ls -lh output/figures/optuna_*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

    else
        echo "⏳ STATUS: STILL RUNNING"
        echo ""
        echo "Progress indicators:"
        # Count completed trials
        completed=$(grep -c "✓ Mean error" optuna_tuning.log 2>/dev/null || echo "0")
        echo "  Completed trials: $completed / 150"

        # Estimate time remaining
        echo "  Estimated completion: ~19:30 today"
    fi
else
    echo "❌ No optuna_tuning.log found - tuning may not have started"
fi

echo ""
echo "================================================================================"
echo "2. CODEBASE CLEANUP STATUS"
echo "================================================================================"

if [ -d "archives/cleanup_20251029_234505" ]; then
    echo "✅ Codebase cleanup completed"
    echo ""
    echo "Cleanup summary:"
    echo "  Archived: $(ls archives/cleanup_20251029_234505/completed_experiments/ 2>/dev/null | wc -l) test scripts"
    echo "  Archived: $(ls archives/cleanup_20251029_234505/duplicate_pinns/ 2>/dev/null | wc -l) duplicate PINNs"
    echo "  Archived: $(ls archives/cleanup_20251029_234505/old_documentation/ 2>/dev/null | wc -l) old docs"
    echo ""
    echo "View full cleanup summary:"
    echo "  cat archives/cleanup_20251029_234505/CLEANUP_SUMMARY.md"
else
    echo "⚠️  Cleanup archive not found"
fi

echo ""
echo "================================================================================"
echo "3. CURRENT PROJECT STRUCTURE"
echo "================================================================================"

echo ""
echo "Production PINNs:"
ls -lh src/multi_option/pinns/*.py 2>/dev/null | grep -E "(optimized_pinn|stabilized_pinn|model|loss|train)" | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Active scripts:"
ls -lh scripts/hyperparameter_tuning/*.py 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh scripts/experiments/*.py 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh scripts/*.py 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Documentation:"
ls -lh docs/*.md README.md 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "================================================================================"
echo "4. RECENT OUTPUT FILES"
echo "================================================================================"

echo ""
echo "Latest figures (last 10):"
ls -lth output/figures/*.png 2>/dev/null | head -10 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'

echo ""
echo "Latest results:"
ls -lth *.json 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'

echo ""
echo "================================================================================"
echo "5. QUICK ACTIONS"
echo "================================================================================"

echo ""
echo "View overnight summary:"
echo "  cat OVERNIGHT_SUMMARY.md"
echo ""
echo "View project structure:"
echo "  cat docs/PROJECT_STRUCTURE.md"
echo ""
echo "View Optuna results (if complete):"
echo "  cat hyperparameter_tuning_optuna_results.json | jq '.best_trial'"
echo "  open output/figures/optuna_comprehensive_results.png"
echo ""
echo "Check full Optuna log:"
echo "  less optuna_tuning.log"
echo ""

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"

if grep -q "OPTIMIZATION COMPLETE" optuna_tuning.log 2>/dev/null; then
    echo ""
    echo "✅ ALL OVERNIGHT WORK COMPLETED!"
    echo ""
    echo "Next steps:"
    echo "  1. Review Optuna results in hyperparameter_tuning_optuna_results.json"
    echo "  2. Check output/figures/optuna_*.png for visualizations"
    echo "  3. If improvement found, update src/multi_option/pinns/optimized_pinn.py"
    echo "  4. Run validation: python scripts/train_pinn.py --validate"
else
    echo ""
    echo "⏳ HYPERPARAMETER TUNING STILL RUNNING"
    echo ""
    echo "Check back later or monitor with:"
    echo "  watch -n 60 'tail -30 optuna_tuning.log'"
fi

echo ""
echo "================================================================================"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
