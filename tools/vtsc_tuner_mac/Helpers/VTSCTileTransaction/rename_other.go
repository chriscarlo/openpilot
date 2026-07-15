//go:build !linux

package main

import "fmt"

func renameExchange(_, _ string) error {
	return fmt.Errorf("renameat2(RENAME_EXCHANGE) is only available in the Linux tici helper")
}
