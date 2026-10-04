"""Explicit fixed/resident/Auto matrix inventory and complete-comparison resume.

Default is dry-run. --execute invokes the existing public Engine drivers only for
eligible complete comparisons. Deferred suites and failed correctness block their
dependencies; legacy campaigns must be explicitly released and attested.
"""
import argparse
import json
import sys
from pathlib import Path
from tools.bench.spec_matrix_resume import ResumeMatrix, MatrixLocked, read_json, atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run',action='store_true')
    action.add_argument('--execute',action='store_true')
    parser.add_argument('--retry-failed',action='store_true',help='explicitly retry a complete comparison whose correctness/driver failed')
    parser.add_argument('--suite',action='append',help='execute only this suite; dependencies remain fully inspected')
    parser.add_argument('--report',type=Path,help='atomic JSON inventory/report path; no checkpoint writes in dry-run')
    args=parser.parse_args(argv)
    if not args.plan.is_absolute():parser.error('plan path must be absolute')
    try:
        matrix=ResumeMatrix(read_json(args.plan))
        if args.suite and any(s not in {v['id'] for v in matrix.plan['suites']} for s in args.suite):raise ValueError('unknown suite selection')
        report=matrix.describe(args.retry_failed)
        if args.execute:
            inventory={r['unit']:r for r in report['units']}
            outcomes=[]
            for unit in matrix.execution_units():
                if args.suite and unit['suite']['id'] not in args.suite:continue
                current=inventory[unit['id']]
                dependencies=unit['suite']['depends_on']
                blocked=any(r['status']!='skip' for r in inventory.values() if r['suite'] in dependencies)
                if blocked or unit['suite']['state']=='deferred':
                    outcomes.append(current);continue
                # Initial dependency blocks are stale after parents completed;
                # run_unit revalidates only this comparison under both locks.
                if current['status']=='skip' or current['eligible_to_execute'] or current['status']=='blocked':
                    current=matrix.run_unit(unit,args.retry_failed)
                    inventory[unit['id']]=current
                outcomes.append(current)
                if current['status']=='failed':break
            report=matrix.describe(args.retry_failed);report['execution_outcomes']=outcomes
        report['mode']='execute' if args.execute else 'dry_run'
        if args.report:atomic_json(args.report,report)
        print(json.dumps(report,indent=2,allow_nan=False))
        return 1 if any(r['status']=='failed' for r in report['units']) else 0
    except (OSError,ValueError,TypeError,KeyError,MatrixLocked) as error:
        print('matrix: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
