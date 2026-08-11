project(
  'skyplot',
  version: run_command('sh', '-c', 'cat VERSION', check: true).stdout().strip(),
)

py = import('python').find_installation(pure: true)

subdir('skyplot')
