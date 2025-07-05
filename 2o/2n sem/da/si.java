class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode res;
        while (list1 == null || list2 == null) {
            if (list1.value < list2.value) {
                res.value = list1.value;
                list1 = list1.next;
            }
            else {
                res.value = list2.value;
                list2 = list2.next;
            }
            ListNode newnode;
            newnode.next = res;
            res = newnode;
            }
    }
    }
}