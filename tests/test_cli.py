"""Tests for CLI commands."""
from click.testing import CliRunner
from pyspectrafuse.pyspectrafuse import cli


class TestCLI:
    """Test cases for CLI commands."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'convert-dat' in result.output.lower() or 'msp' in result.output.lower()

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower()

    def test_convert_dat_command_help(self):
        """Test convert-dat command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert-dat', '--help'])
        assert result.exit_code == 0
        assert 'parquet' in result.output.lower() or 'dat' in result.output.lower()

    def test_msp_command_help(self):
        """Test msp command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['msp', '--help'])
        assert result.exit_code == 0
        assert 'msp' in result.output.lower() or 'parquet' in result.output.lower()

    def test_convert_dat_command_missing_args(self):
        """Test convert-dat command with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert-dat'])
        assert result.exit_code != 0 or '--parquet_dir' in result.output

    def test_msp_command_missing_args(self):
        """Test msp command with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ['msp'])
        assert result.exit_code != 0 or '--parquet_dir' in result.output

    def test_incremental_help(self):
        """The incremental group now exposes only extract-reps-dat."""
        runner = CliRunner()
        result = runner.invoke(cli, ['incremental', '--help'])
        assert result.exit_code == 0
        assert 'extract-reps-dat' in result.output
        # merge-clusters was folded into `build-cluster-db --existing_metadata`.
        assert 'merge-clusters' not in result.output

    def test_incremental_extract_reps_dat_help(self):
        """Test incremental extract-reps-dat command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['incremental', 'extract-reps-dat', '--help'])
        assert result.exit_code == 0
        assert 'cluster_metadata' in result.output

    def test_build_cluster_db_help_mentions_merge_flags(self):
        """build-cluster-db should advertise --existing_metadata / --existing_membership."""
        runner = CliRunner()
        result = runner.invoke(cli, ['build-cluster-db', '--help'])
        assert result.exit_code == 0
        assert '--existing_metadata' in result.output
        assert '--existing_membership' in result.output

    def test_convert_mzml_dat_help(self):
        """Test ID-free mzML conversion command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert-mzml-dat', '--help'])
        assert result.exit_code == 0
        assert 'mzml' in result.output.lower()
        assert '--workers' in result.output

    def test_filter_noid_clusters_help(self):
        """Test ID-free cluster quality command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['filter-noid-clusters', '--help'])
        assert result.exit_code == 0
        assert 'quality' in result.output.lower()

    def test_build_noid_cluster_db_help(self):
        """Test ID-free DB builder command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['build-noid-cluster-db', '--help'])
        assert result.exit_code == 0
        assert 'no-id' in result.output.lower()

    def test_msp_noid_help(self):
        """Test ID-free MSP command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['msp-noid', '--help'])
        assert result.exit_code == 0
        assert 'msp' in result.output.lower()

    def test_legacy_noid_maracluster_command_is_not_public(self):
        """Standalone no-ID MaRaCluster execution is no longer a public entry."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'run-noid-maracluster' not in result.output
